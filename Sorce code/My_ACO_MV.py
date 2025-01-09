import numpy as np


def ACO_MV_R(pop_x, len_r, kesi, cum_p,  dn_r, up_r, K):       #生成类变量子代
    #pop_x：初代的连续变量。  len_r：连续变量的数量。 kesi：搜索宽度。cum_p：累积概率数组，用于轮盘赌选择。dn_r 和 up_r：实数变量的下界和上界  K：K个最好解。
    n_z = np.zeros([1, len_r])    #创建新数组储存子代
    # pop_size = K
    # 计算SA中每个个体的权重
    # w = (1 / (q * pop_size * np.pi)) \
    #     * np.exp(-((pop_rank-1) ** 2) / (2 * (q * pop_size) ** 2)) # the original paper is sqrt(2)
    #根据概率p轮盘赌选择一个个体
    idx_gr = RW_Select(cum_p)            #选出被选的索引  存入idx_gr
    # rnd = np.random.rand(1)
    # idx_gr = np.where(cum_p>=rnd)
    z, B, x0 = ROT(pop_x, idx_gr, len_r)          # 输入种群的x，
    #返回相乘结果，轮盘赌所选解与其他解相减得到矩阵分解的正交矩阵，再次选择的一个解       返回旋转后的矩阵 z，基矩阵 B 和偏移向量 x0

    for j in range(0, len_r):
        mu = z[idx_gr, j]           #轮盘都选出的索引    选中个体在第 j 个维度的值
        sigma = kesi * np.sum(abs(mu - z[:, j])) / (K - 1)        #论文中11式             abs绝对值

        n_z[0, j] = mu + sigma * np.random.randn(1)         #？？？n_z[0, j] 是新生成的候选解在第 j 个维度的值，它是通过在正态分布 N(mu, sigma) 中抽取一个随机数来生成的
    x_r = np.dot(n_z, B.T) + x0          #使用基矩阵 B 和偏移向量 x0 重建新的候选解 x_r。这里 n_z 是通过旋转操作和正态分布生成的新值
    x_r = Repair(x_r, dn_r,up_r)           #将x_r控制在范围内
    return x_r       #输出连续变量子代


def ROT(x_r, idx_gr, len_r):
    #连续变量     轮盘赌选中的索引    连续变量的数量
    flag = (np.sum(np.sum(x_r - x_r[idx_gr, :])) != 0) & (len_r > 1)   #计算布尔值，在选中的索引和输入的连续变量不相同，且长度大于1，若长度等于1，旋转则没有意义。
    if flag:
        B = VCH(x_r, x_r[idx_gr, :])         #将连续变量与所选变量生成矩阵     B是轮盘赌所选解与其他解相减得到矩阵分解的正交矩阵
    else:
        B = np.eye(len_r)
        #np.eye(len_r)         生成一个行数为len_r的单位矩阵

    if np.linalg.matrix_rank(B) != len_r:             #计算B的秩（rank）  检查VCH返回是否满秩
        B = np.eye(len_r)

    z_r = np.dot(x_r - x_r[idx_gr, :], B)         #x_r中与选中的变量相减，再与B点积，
    x0 = x_r[idx_gr, :]                       #选中个体赋给x0
    return z_r, B, x0               #返回相乘结果，轮盘赌所选解与其他解相减得到矩阵分解的正交矩阵，再次选择的一个解


# 生成旋转矩阵
def VCH(s, s1):             #    s为一个连续变量  s1为轮盘赌选中的那个解      ？？？？？？？？？？
    n = np.size(s, 1)         #计算s的列数，即一个解的维度
    A = np.zeros([n, n])      #创建一个矩阵
    for i in range(0, n):
        ds = np.sqrt(np.sum((s1[i:n] - s[:, i:n]) ** 2, 1))        #计算 s1 从第 i 行开始的子数组与 s 从第 i 列开始的子数组之间的欧几里得距离，并对这些距离求平方根   论文中11式
        #计算
        p = (ds ** 4) / np.sum(ds ** 4)       #计算一个概率分布 p，它是距离的四次方除以所有四次方距离的和    ？？？？
        idx = RW_Select(p)            # 轮盘赌选一个概率的索引
        # for j in range(np.size(p, 0)):
        #     # p[j] = w[j] / np.sum(w)
        #     if np.sum(p[:j]) >= np.random.rand(1):
        #         idx = j
        #         break
        # idx = Functional_Operator.RW_Select(p)  ####################
        A[i, :] = s1 - s[idx, :]    #将s中每个解选出的与s1做差生成矩阵A
        s = np.delete(s, idx, axis=0)    #删除s中idx所指的行

    if np.max(A) < 1e-5:             #检查矩阵  最大值是否小于一个很小的阈值
        B, non = np.linalg.qr(np.random.random([n, n]))      #使用一个随机矩阵进行QR分解，并将结果赋值给B和non
        #QR分解 Q是正交矩阵，矩阵R是上三角矩阵
    else:
        B, non = np.linalg.qr(A.T)          #对矩阵A的转置进行QR分解，并将结果赋值给B和non
    return B                   #返回对A矩阵的正交矩阵

def Repair(x_r, dn_r, up_r):
    for i in range(np.size(x_r, 0)):
        for j in range(np.size(x_r, 1)):
            if x_r[i,j] < dn_r or x_r[i,j] > up_r:
                x_r[i, j] = dn_r + np.random.rand(1)*(up_r - dn_r)

    return x_r         #将解中不符合上下界的解重新生成一个随机的解

def RW_Select(p_set):              #接受一个概率的集合
    lvalue = np.random.rand(1)           #生成随机数
    probability = 0
    for i in range(np.size(p_set, 0)):
        probability += p_set[i]
        if probability >= lvalue:       #循环遍历概率数组
                idx = i
                break
    return idx                 #返回选到的索引

#################################
#     函数：ACO_MV离散算子       #
#################################
def ACO_MV_C(pop_x, len_c, w, q, N_lst, v_dv):
    #输入初代种群x   类变量长度  权重数组   最佳解影响因子    类变量的可能值的数量列表   类变量可能值的矩阵
    x_c = np.zeros([1, len_c])     #用于存储新生成的值
    for j in range(0, len_c):
        pl = Cal_pl(pop_x[:, j], N_lst[j], v_dv[j], w, q)   #计算每个值概率
        idx_gc = RW_Select(pl)      #轮盘赌选一个索引
        x_c[0, j] = v_dv[j, idx_gc]   #将选择的类别变量值赋给 x_c[0, j]
    x_c = x_c.astype(float)     #将 x_c 转换为浮点数类型。
    return x_c      #完成类变量自带生成

# 更新类别变量集合中每个元素可能被选择的概率
def Cal_pl(x_c, l, v_dv, w, q):
    #x_c：种群的类别变量数据   l：类别变量的可能值的数量   v_dv：类别变量的可能值    w：权重数组    q：最佳解的影响因子
    u = np.zeros(l)         #初始化数组，用于存储每个值出现次数
    wjl = np.zeros(l)       #用于存储每个值权重
    wl = np.zeros(l)        #用于存储归一化的权重
    for i in range(0, l):
        idx_l = (x_c == v_dv[i])
        u[i] = np.sum(idx_l)        #计算种群中等于当前值 v_dv[i] 的次数，并存储在 idx_l 中

        if np.sum(idx_l) == 0:     #检查当前值是否在种群中没有出现
            wjl[i] = 0            #如果没有出现，则将 wjl[i] 设置为0。
        else:
            wjl[i] = np.max(w[idx_l])       #将当前值出现的所有解的最大权重存储在 wjl[i] 中

    eta = 100 * np.sum(u == 0)         #计算未出现值的比例 eta。
    for i in range(0, l):                     #实现论文中13式
        if (eta > 0) & (u[i] > 0):
            wl[i] = wjl[i] / u[i] + q / eta
        elif (eta == 0) & (u[i] > 0):
            wl[i] = wjl[i] / u[i]
        elif (eta > 0) & (u[i] == 0):
            wl[i] = q / eta

    out = wl / np.sum(wl)    #归一化权重，确保所有权重之和为1
    return out     #输出概率分布

#################################
def ACO_MV_generates(database, len_r, len_c, dn_r, up_r, N_lst, v_dv):
                #len_r（实数变量的数量）、len_c（类别变量的数量）、dn_r（实数变量的下界）、up_r（实数变量的上界）、N_lst（可能与类别变量相关的列表）、v_dv（可能与变异或其他操作相关的参数
    K = 60                   #文中提到的K个已评估的最好解
    M = 100
    pop_x = database[0][:K]      #数据库前60个解
    pop_y = database[1][:K]      #前60目标值
    q = 0.05099 # Influene of the best-quality solutions in ACOmv       最佳解的影响因子
    kesi= 0.6795 # Width of the search in ACOmv                  搜索宽度
    x_r_generate = np.zeros((M,len_r))       #创建数组放连续变量和类变量
    x_c_generate = np.zeros((M,len_c))
    w = np.zeros(K)
    p = np.zeros(K)
    cum_p =np.zeros(K)                    #长度60三个数组
    for j in range(0,K):
        pop_rank = j + 1              #文中计算权重的rank。
        w[j] = (1 / (q * K * np.pi)) \
            * np.exp(-((pop_rank - 1) ** 2) / (2 * (q * K) ** 2))  # the original paper is sqrt(2)             根号2pi  分之一          文中的8式，算各好解的权重

    for j in range(0,K):
        p[j] = w[j] / np.sum(w)         #根据权重算轮盘赌的概率
        cum_p[j] = np.sum(p[:j])          # 算前j个概率和。。方便轮盘赌的选择，直接超过第几个概率和就取那个概率所对应的解

    for i in range(0,M):
        # pop_rank = i+1
        x_r_generate[i, :] = ACO_MV_R(pop_x[:,:len_r], len_r, kesi, cum_p, dn_r, up_r, K)
        x_c_generate[i, :] = ACO_MV_C(pop_x[:, len_r:], len_c,  w, q, N_lst, v_dv)             #生成子代的方法

    return x_r_generate, x_c_generate           #返回生成的子代
#################################