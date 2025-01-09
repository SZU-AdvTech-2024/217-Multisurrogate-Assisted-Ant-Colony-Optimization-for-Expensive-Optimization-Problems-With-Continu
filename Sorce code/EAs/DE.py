import numpy as np


class Pop:
    def __init__(self, X, func):
        self.X = X
        self.func = func
        self.ObjV = None

    def __add__(self, other):
        self.X = np.vstack([self.X, other.X])   #沿行方向堆叠 行方向堆叠
        self.ObjV = np.hstack([self.ObjV, other.ObjV])         #水平堆叠   列方向堆叠
        return self

    def cal_fitness(self):  # 计算目标值   计算适应度 实例方法
        self.ObjV = self.func(self.X)


class DE(object):
    def __init__(self, func, max_iter, dim, lb, ub, initX=None):          #func要优化的函数 ，max_iter 最大迭代次数，dim  维度，lb,ub 上下界   initX初始种群

        self.max_iter = max_iter
        self.initX = initX
        if (self.initX is None):
            self.popsize = 100       #没有初始种群，将种群大小设为100
        else:
            self.popsize = self.initX.shape[0]       #

        self.F = 0.5             #缩放因子，用于控制种群中随机选择的两个个体之间的差值的规模。
        self.CR = 0.8            #差分进化算法中的交叉概率 CR，这里设置为 0.8。交叉概率决定了在生成新个体时，保留父代个体的哪些部分。

        self.func = func
        self.dim = dim
        self.xmin = np.array(lb)
        self.xmax = np.array(ub)

        self.xbest = None
        self.ybest = None
        self.pop = None

    def initPop(self):                                        #初始化
        X = np.zeros((self.popsize, self.dim))
        area = self.xmax - self.xmin
        for j in range(self.dim):
            for i in range(self.popsize):
                X[i, j] = self.xmin[j] + np.random.uniform(i / self.popsize * area[j], (i + 1) / self.popsize * area[j])   #随机生成浮点数
                '''
                numpy.random.uniform(low=0.0, high=1.0, size=None)
                low: （可选）生成的随机数的下限，默认为 0.0。
                high: （可选）生成的随机数的上限，默认为 1.0。
                size: （可选）输出随机数的形状。如果未提供，返回一个标量；如果提供，则返回一个数组。'''
            np.random.shuffle(X[:, j])       #对第 j 维度的所有个体进行随机打乱，以确保种群在这个维度上的多样性。

        self.pop = Pop(X, self.func)              #创建一个 Pop 类的实例 self.pop，传入初始化的种群 X 和目标函数 self.func。这里假设 Pop 是一个处理种群的类，它可能有方法来计算适应度、选择、交叉和变异等。
        self.pop.cal_fitness()            #计算适应度，为进化做准备

    def mutation(self):                                       #变异
        muX = np.empty((self.popsize, self.dim))             #numpy.empty(shape, dtype=float, order='C')  'C' 表示按行优先（C-style），'F' 表示按列优先（Fortran-style）   编译后数据不初始化为0，不用zeros
        b = np.argmin(self.pop.ObjV)                      #ObjV适应度
        for i in range(self.popsize):  # DE/rand/1         使用一个差异向量
            r1 = r2 = r3 = 0
            while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2:
                r1 = np.random.randint(0, self.popsize - 1)
                r2 = np.random.randint(0, self.popsize - 1)
                r3 = np.random.randint(0, self.popsize - 1)                   #初始化三个不同的随机数，r1,2,3

            mutation = 0.5 * self.pop.X[b] + 0.5 * self.pop.X[r1] + self.F * (self.pop.X[r2] - self.pop.X[r3])       #变异向量=0.5*好解+0.5*r1+缩放因子F*（r2-r3）

            for j in range(self.dim):
                #  判断变异后的值是否满足边界条件，不满足需重新生成
                if self.xmin[j] <= mutation[j] <= self.xmax[j]:
                    muX[i, j] = mutation[j]
                else:
                    rand_value = self.xmin[j] + np.random.random() * (self.xmax[j] - self.xmin[j])
                    muX[i, j] = rand_value
        return muX             #最后返回变异种群，数量和初始种群一样

    def crossover(self, muX):                           #交叉
        crossX = np.empty((self.popsize, self.dim))
        for i in range(self.popsize):
            rj = np.random.randint(0, self.dim - 1)       #随机整数，交叉过程基准点
            for j in range(self.dim):
                rf = np.random.random()   #随机生成浮点数
                if rf <= self.CR or rj == j:                #随机生成小于交叉概率0.8，或维度为随机生成的基准点
                    crossX[i, j] = muX[i, j]                     #变异值赋给crossX
                else:
                    crossX[i, j] = self.pop.X[i, j]                #不满足不赋crossX，交叉

        return crossX

    def selection(self, crossPop):
        for i in range(self.popsize):
            if crossPop.ObjV[i] < self.pop.ObjV[i]:
                self.pop.X[i] = crossPop.X[i]
                self.pop.ObjV[i] = crossPop.ObjV[i]
                # self.pop.CV[i] = crossPop.CV[i]

    def update_best(self):
        rank = np.argsort(self.pop.ObjV)
        self.xbest = self.pop.X[rank[0]]
        self.ybest = self.pop.ObjV[rank[0]]

    def run(self):
        if self.initX is None:
            self.initPop()
        else:
            self.pop = Pop(self.initX, self.func)
            self.pop.cal_fitness()
        self.update_best()

        for i in range(self.max_iter):
            muX = self.mutation()
            crX = self.crossover(muX)

            epop = Pop(crX, self.func)
            epop.cal_fitness()

            self.selection(epop)
            self.update_best()

        return self.xbest

