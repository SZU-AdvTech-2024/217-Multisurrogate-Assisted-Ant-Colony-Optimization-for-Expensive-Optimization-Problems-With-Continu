import numpy as np
import LSBT
import RBF
import RBF2
from pyDOE import *   #pyDOE是一个用于实验设计的Python库
# import RBF_LS
from My_ACO_MV import ACO_MV_generates
from scipy.optimize import minimize  #提供了多种优化算法
from functools import partial   #partial函数用于固定一个或多个函数参数
from EAs.DE import DE
from smt.surrogate_models import rbf  #smt.surrogate_models模块提供了多种代理模型
# Algorithm class
class MiACO(object):
    def __init__(self, maxFEs, popsize, dim, clb, cub, N_lst, v_dv, prob, r, database=None):
        '''
        :param maxFEs: the maximum number of function evaluations
        :param popsize: the size of population
        :param dim: the dimension of decision variables
        :param clb: the lower bounds of continuous decision variables
        :param cub: the upper bounds of continuous decision variables
        :param N_lst: the number of values for discrete variables
        :param prob: an expensive mixed-variable optimization problem instance
        :param r: the dimension of continuous decision variables
        v-dv应该是
        '''
        self.maxFEs = maxFEs
        self.popsize = popsize

        self.dim = dim
        self.cxmin = np.array(clb)
        self.cxmax = np.array(cub)
        self.dn_r = self.cxmin[1]
        self.up_r = self.cxmax[1]

        self.N_lst = N_lst
        self.v_dv = v_dv

        self.prob = prob
        self.r = r
        self.o = self.dim - self.r  #决策变量维度-连续决策变量维度 = 类变量维度
        self.c_result = []
        # surrogate models
        self.global_sm = None
        self.local_sm1 = None
        self.local_sm2 = None
        self.sm3 = None

        self.pop = None
        self.database = None
        self.init_size = popsize  # the size of initial samples
        self.gen = None

        self.xbest = None
        self.ybest = None
        self.ybest_lst = []

        self.data = None
        self.melst = []

    def initPop(self):    #initpop初始化种群
        X_o = np.zeros((self.init_size, self.o))  #初始化 初始样本长度（popsize行，1行o个数）
        inity = np.zeros((self.init_size))     #初始popsize个数
        area = self.cxmax - self.cxmin    #连续变量的区域
        LBI = np.tile(self.cxmin, (self.init_size, 1))  #numpy.tile(A, reps)A 是要重复的数组。 reps 是一个整数或整数元组，指定了数组 A 应该在每个维度上重复的次数。
        UBI = np.tile(self.cxmax, (self.init_size, 1))    #用这样控制整个种群的上下界，上下界变化只需修改cxmin
        # continuous part

        # X_r = LBI[1, 1] + (lhs(self.r, samples=self.init_size, criterion='center') * (UBI[1, 1] - LBI[1, 1]))
        X_r = LBI[1, 1] + (lhs(self.r, samples=self.init_size) * (UBI[1, 1] - LBI[1, 1])) #lhs()生成了一个样本数为init_size，维度为r的种群矩阵
        '''X_r代表初始化的连续变量部分的种群，其中每个个体的每个维度都是通过LHS生成的，并且调整到相应的上下界范围内。
        这样的初始化有助于确保种群在搜索空间内均匀分布，从而提高优化算法的性能和收敛性。
        LBI[1,1] UBI[1,1]均为上下界， 用 LHS生成的数∈（0，1） * area + 下界  得到 上下界中的随机数 
        
        至此，连续变量初始化完成，初始种群矩阵保存在X_r中'''
        for j in range(self.o):                   #选类变量
            for i in range(self.init_size):
                v_ca = self.v_dv[j]
                X_o[i, j] = v_ca[np.random.randint(self.N_lst[j])]   #self.N_lst[j]  离散变量的数量  np.random.randint生成随机整数
        X = np.concatenate((X_r, X_o), axis=1)    #numpy.concatenate 是一个用于沿指定轴连接多个数组的函数  axis决定沿哪个维度连接数组
        for j in range(self.init_size):
            inity[j] = self.prob(X[j, :])  #prob是昂贵问题的例子？传递到数组inity

        for j in range(self.init_size):
            if j == 0:
                self.c_result.append(np.min(inity[j]))  #将inity最小值加入c.result
            else:
                self.c_result.append(np.min(inity[0:j]))    #j=1时，将inity 中前j-1个中最小值赋值给c_result
        self.gen = self.init_size
        inds = np.argsort(inity)   #给按inity中，值从小到大排列的索引
        self.database = [X[inds], inity[inds]]  #将初始化种群按prob从小到大放进数据库
        self.data = [X[inds], inity[inds]]
        self.xbest = self.database[0][0]
        self.ybest = self.database[1][0]        #最好解是datebase的第一个元素，最好目标函数值是最好的prob,,,那prob是目标函数？ 标记prob是昂贵问题实例


    # Multi surrogate-assisted selection
    def MS_Selection_Rand(self, x_r_generate, x_c_generate):

        candidate_set = np.zeros((3, self.dim))
        #candidate_set =[]

        surr_lsbt = LSBT.Create(self.database, self.r, self.o, self.dn_r, self.up_r, self.N_lst, self.v_dv)
        surr_rbf = RBF.Create(self.v_dv, self.database, self.r, self.o, self.dn_r, self.up_r)
        ###################################################################
        # 代理模型预选择
        ###################################################################
        # 基于LSBT的预选择
        f_lsbt = LSBT.Predict(surr_lsbt, x_r_generate, x_c_generate, self.v_dv)
        index = np.argmin(f_lsbt)        #    下午开始，。。argmin用来找f_lsbt中最小     表示预测误差最小的点
        Off_lsbt = np.concatenate((x_r_generate[index, :], x_c_generate[index, :]))   #按规定轴链接，没axis参数默认沿第一个轴连   把输入最好解连续变量类变量连接

        candidate_set[0] = Off_lsbt    #将Xbest放在创建候选数组第一个
        # x_r_generate[index, :] = None
        # x_c_generate[index, :] = None
        x_r_generate = np.delete(x_r_generate, index, axis=0)
        x_c_generate = np.delete(x_c_generate, index, axis=0)          #输入的解中删除最好解
        # candidate_set = np.append(candidate_set, Off_lsbt, axis=0)
         # 基于RBF的预选择
        f_lrbf = RBF.Predict_one(surr_rbf, x_r_generate, x_c_generate, self.r, self.o)    #得到RBF的预测值

        index = np.argmin(f_lrbf)
        Off_rbf = np.concatenate((x_r_generate[index, :], x_c_generate[index, :]))
        candidate_set[1] = Off_rbf            #候选解集第二个放RBF的解
        x_r_generate = np.delete(x_r_generate, index, axis=0)                 #在解中删掉候选的解
        x_c_generate = np.delete(x_c_generate, index, axis=0)
        # candidate_set = np.append(candidate_set, Off_rbf, axis=1)

        # 随机预选择
        index = np.random.randint(0, np.size(x_r_generate, 0))
        Off_rand = np.concatenate((x_r_generate[index, :], x_c_generate[index, :]), axis=0)
        candidate_set[2] = Off_rand
        # candidate_set = np.append(candidate_set, Off_rand, axis=1)

        return candidate_set       #随机选一个解，选完还是删

    #Local surrogate for local search
    def SALS(self, X_r, y_r):
        index = np.argmin(y_r)      #找到最优解
        X_r_best = X_r[index, :]        #根据索引找到最好解
        data_2 = [X_r, y_r]           #创建列表
        # local_sat = np.concatenate(data_2, self.r, self.o, self.dn_r, self.up_r)
        surr_rbfs = RBF2.Create(data_2, self.r, self.o, self.dn_r, self.up_r)

        def obj_cons_func(x, surr_rbf):
            y = RBF2.Predict_one(surr_rbf, x)
            return y

        obj = partial(obj_cons_func, surr_rbf=surr_rbfs)           #functools.partial 来固定 obj_cons_func 函数的 surr_rbf 参数为 surr_rbfs，创建一个新的函数 obj。？？？？？？？？？？？？？？？？
        bounds = ()       #用于定义边界
        # X_r_best = np.reshape(1, -1)
        for i in range(0, self.r):
            bounds = bounds + ((self.dn_r, self.up_r),)      #为每个元组添加一个边界
        # res = minimize(obj, X_r_best, method='SLSQP', options={'maxiter':3000})
        res = minimize(obj, X_r_best, method='SLSQP', bounds = bounds)      #使用 scipy.optimize.minimize 函数，以 obj 作为目标函数，X_r_best 作为初始点，SLSQP 作为优化方法，bounds 作为变量的边界，执行优化过程
        # return res.x

        self.sm3 = rbf.RBF(print_global=False)      #创建一个 RBF 模型实例 self.sm3，并设置 print_global 参数为 False。
        self.sm3.set_training_values(X_r, y_r)       #为 self.sm3 设置训练数据，即 X_r 和 y_r。
        self.sm3.train()                            #训练 self.sm3 模型    调用RBF训练

        ga = DE(max_iter=30, func=self.sm3.predict_values, dim=self.r, lb=self.cxmin, ub=self.cxmax,
                initX=X_r)                                #差分进化？  进化得子代
        X_l = res.x
        #X_l = ga.run()             #执行ga    挑最好的解  存贮在x_l
        print("local search")

        return np.append(X_l, self.xbest[self.r:]).reshape(1, -1)        #将 X_l 与 self.xbest 中从索引 self.r 开始的部分连接起来，并重塑为一个一行的数组，然后返回这个数组。

    def data_selection(self):
        best_c = self.xbest[self.r:]
        inds = []
        for i in range(len(self.database[1])):        #database 1 应该是目标值，database0 是解
            if (np.all(self.database[0][i, self.r:] == best_c)):    #检查目标值中第i行从索引self.r开始的部分是否与best_c完全相等。np.all函数用于检查数组中的所有元素是否都为True。
                inds.append(i)     #将索引存入inds    在数据库中找到

        X_r = self.database[0][inds, :self.r]  #中选取索引在inds中的行，并且只选择这些行的前self.r个特征，将这些数据赋值给X_r。
        y_r = self.database[1][inds]     #目标变量
        size = len(y_r)

        return size, X_r, y_r

        # update the database and current population
    def update_database(self, X, y):
        self.data[0] = np.append(self.data[0], X)
        self.data[1] = np.append(self.data[1], y)    #将X,Y分别加入database

        size = len(self.database[1])       #y的规模
        for i in range(size):
            if (self.database[1][i] > y):        #如果输入的y比数据库第i个值更好，插入数据库
                self.database[0] = np.insert(self.database[0], i, X, axis=0)
                self.database[1] = np.insert(self.database[1], i, y)
                break

        # self.pop.X = self.database[0][:self.popsize]
        # self.pop.realObjV = self.database[1][:self.popsize]
        self.xbest = self.database[0][0]               #更新最好的值
        self.ybest = self.database[1][0]

    #mail loop
    def run(self):
        if self.database is None:
            self.initPop()                 #初始化
        else:

            initX = self.database[0]
            inity = self.database[1]
            inds = np.argsort(inity)              #从小排序列

            self.data = [initX[inds], inity[inds], self.database[2][inds]]
            self.database = [initX[inds], inity[inds]]             #按顺序输入到data和database

            self.xbest = self.database[0][0]
            self.ybest = self.database[1][0]
            self.gen = len(self.database[1])                 #目标值数量

        while (self.gen < self.maxFEs):            #目标值数量小于最大评估次数
            x_r_generate, x_c_generate = ACO_MV_generates(self.database, self.r, self.o, self.dn_r, self.up_r, self.N_lst, self.v_dv)
            # offspring = np.concatenate(x_r_generate, x_c_generate, axis=1)
            # prescreening  search
            candidate_set_sum = []
            candidate_set = self.MS_Selection_Rand(x_r_generate, x_c_generate)
            candidate_set_sum = candidate_set
            # Local continuous search
            size, X_r, y_r = self.data_selection()

            # if (size > 5 * self.r):
            if (size > 5 * self.r):
                candidate_set2 = self.SALS(X_r, y_r)
                candidate_set_sum = np.concatenate((candidate_set_sum, candidate_set2), axis=0)

            # evaluate candidates
            for i in range(np.size(candidate_set_sum, 0)):
                total = ['LSBT-prescreen', 'RBF-prescreen','Random-prescreen', 'RBF-local search']
                dx = np.abs((candidate_set_sum[i,:]) - (self.database[0]))
                meadis = np.min(np.mean(dx, 1))
                if meadis > 1e-4:
                    y = self.prob(candidate_set_sum[i,:])
                    self.update_database(candidate_set_sum[i,:], y)
                    self.gen += 1
                    if self.gen > self.maxFEs:
                        break
                    else:
                        self.ybest_lst.append(self.ybest)
                        self.c_result.append(self.ybest)
                        print("{}/{} gen: {} {}".format(self.gen, self.maxFEs, y, total[i]))

        return self.xbest, self.ybest, self.ybest_lst, self.database,  self.c_result
