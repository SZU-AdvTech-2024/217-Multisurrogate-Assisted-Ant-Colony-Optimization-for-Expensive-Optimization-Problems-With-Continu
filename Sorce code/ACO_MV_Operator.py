#################################
#     函数：ACO_MV连续算子       #
#################################
# Par_ACO_MV.q
# Par_ACO_MV.kesi
def ACO_MV_R(Pop,Par_ACO_MV):
    n_z = numpy.zeros([1,Pop.len_r])
    #计算SA中每个个体的权重
    w = ( 1/(Par_ACO_MV['q']*Pop.pop_size*numpy.pi) )\
        *numpy.exp( -((Pop.pop_rank)**2)/(2*(Par_ACO_MV['q']*Pop.pop_size)**2) )
    p = w/numpy.sum(w)
    #根据概率p轮盘赌选择一个个体
    idx_gr = Functional_Operator.RW_Select(p)
    z,B,x0 = ROT(Pop.x_r,idx_gr,Pop.len_r)
    for j in range(0,Pop.len_r):
        mu = z[idx_gr,j]
        sigma = Par_ACO_MV['kesi']*numpy.sum( abs( mu - z[:,j] ) )/(Pop.pop_size-1)
        
        n_z[0,j] = mu + sigma*numpy.random.randn(1)
    x_r = numpy.dot(n_z,B.T) + x0
    x_r = Repair(x_r,Pop.up_r,Pop.dn_r)
    return x_r
#执行旋转操作   
def ROT(x_r,idx_gr,len_r):
    flag = (numpy.sum( numpy.sum(x_r - x_r[idx_gr,:]) ) != 0)&( len_r>1 )
    if flag:
        B = VCH(x_r,x_r[idx_gr,:])
    else:
        B = numpy.eye(len_r)
        
    if numpy.linalg.matrix_rank(B) != len_r:
        B = numpy.eye(len_r)
        
    z_r = numpy.dot(x_r - x_r[idx_gr,:],B)
    x0  = x_r[idx_gr,:]
    return z_r, B, x0
#生成旋转矩阵
def VCH(s,s1):
    n = numpy.size(s,1)
    A = numpy.zeros([n,n])
    for i in range(0,n):
        ds = numpy.sqrt( numpy.sum( (s1[i:n] - s[:,i:n])**2,1 ) )
        p  = (ds**4)/numpy.sum(ds**4)
        idx = Functional_Operator.RW_Select(p)
        A[i,:] = s1 - s[idx,:]
        s = numpy.delete(s,idx,axis = 0)
        
    if numpy.max(A)<1e-5:
        B,non = numpy.linalg.qr(numpy.random.random([n,n]))
    else:
        B,non = numpy.linalg.qr(A.T)
    return B
#################################
    

#################################
#     函数：ACO_MV离散算子       #
#################################
# Par_ACO_MV.q
# Par_ACO_MV.kesi
def ACO_MV_C(Pop,Par_ACO_MV):
    x_c = numpy.zeros([1,Pop.len_c])
    #计算SA中每个个体的权重
    w = ( 1/(Par_ACO_MV['q']*Pop.pop_size*numpy.pi) )\
        *numpy.exp( -((Pop.pop_rank)**2)/(2*(Par_ACO_MV['q']*Pop.pop_size)**2) )
    
    for j in range(0,Pop.len_c):
        pl = Cal_pl(Pop.x_c[:,j],Pop.l[j],w,Par_ACO_MV['q'])
        idx_gc = Functional_Operator.RW_Select(pl)
        x_c[0,j] = idx_gc
    x_c = x_c.astype(int)
    return x_c
#更新类别变量集合中每个元素可能被选择的概率
def Cal_pl(x_c,l,w,q):
    u = numpy.zeros(l)
    wjl = numpy.zeros(l)
    wl  = numpy.zeros(l)
    for i in range(0,l):
        idx_l = ( x_c==i )
        u[i]  = numpy.sum(idx_l)
        
        if numpy.sum(idx_l) == 0:
            wjl[i] = 0
        else:
            wjl[i] = numpy.max( w[idx_l] )
            
    eta = 100*numpy.sum( u==0 )
    for i in range(0,l):
        if (eta>0)&(u[i]>0):
            wl[i] = wjl[i]/u[i] + q/eta
        elif (eta==0)&(u[i]>0):
            wl[i] = wjl[i]/u[i]
        elif (eta>0)&(u[i]==0):
            wl[i] = q/eta
            
    out = wl/numpy.sum(wl)
    return out
#################################


#################################
#  函数：ACO_MV算子生成一组后代  #
#################################
def ACO_MV_generates(Pop,Par_ACO_MV): 
    x_r_generate = numpy.zeros([Par_ACO_MV['m'],Pop.len_r])
    x_c_generate = numpy.random.randint(0,1,[Par_ACO_MV['m'],Pop.len_c])
    for i in range(0,Par_ACO_MV['m']):
        x_r_generate[i,:] = ACO_MV_R(Pop,Par_ACO_MV)
        x_c_generate[i,:] = ACO_MV_C(Pop,Par_ACO_MV)
    return x_r_generate, x_c_generate
#################################