# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 09:16:40 2020

@author: Katyusha
"""

import numpy
from EA import Functional_Operator
from functools import partial
from scipy.optimize import minimize

#################################
#   函数：RBF-based 局部搜索     #
#################################
def RBF_LS(DB,x0):
    bounds = ()
    len_x = DB.len_r
    for i in range(0,len_x):
        bounds = bounds + ((DB.dn_r[i],DB.up_r[i]),)
    
    if DB.len_f == 1:
        surr_rbfs = Create(DB)
        obj = partial(obj_cons_func, surr_rbf=surr_rbfs)        
        res = minimize(obj, x0, method='SLSQP')
    else:
        surr_rbfs = Creates(DB)
        obj = partial(obj_cons_func, surr_rbf=surr_rbfs[0])  
        cons = ()
        for i in range(1,DB.len_f):
            cons_add_func = partial(obj_cons_func, surr_rbf=surr_rbfs[i]) 
            cons_add = {'type': 'ineq', 'fun': cons_add_func}
            cons = cons + (cons_add,)
        res = minimize(obj, x0, method='SLSQP',constraints=cons,bounds = bounds)
    out_f = res.fun
    out_x = numpy.array([res.x])
    return out_f, out_x
#################################     


#########################################
# 函数：调整格式后的目标函数/约束条件     #
#########################################
def obj_cons_func(x,surr_rbf):
    y = Predict(surr_rbf,x)
    return y
#################################
    

#################################
#     函数：创建RBF代理模型      #
#################################
def Create(DB,t = 0):
    ax = (DB.x_r - DB.dn_r)/(DB.up_r - DB.dn_r) - 0.5    
    ay = DB.f[:,t]
    ymin = numpy.min(ay)
    ymax = numpy.max(ay)
    ay = 2*(ay - ymin)/(ymax - ymin) - 1
    
    dis_euc = numpy.zeros([DB.db_size,DB.db_size])
    for i in range(0,DB.db_size):
        dis_euc[i,:] = Functional_Operator.Euclidean_Dis(ax,ax[i,:])
    
    r = dis_euc
    Phi = r**3
    
    PPhi = numpy.dot(Phi.T,Phi)
    PPhinv = numpy.linalg.pinv(PPhi)
    Phiy = numpy.dot(Phi.T, ay)
    theta = numpy.dot(PPhinv,Phiy)
    
    surr_rbf = {"alpha":theta}
    surr_rbf.update({"ymin":ymin})
    surr_rbf.update({"ymax":ymax})
    
    surr_rbf.update({"ax":ax})
    surr_rbf.update({"up_r":DB.up_r})
    surr_rbf.update({"dn_r":DB.dn_r})
    surr_rbf.update({"Phi":Phi})
    return surr_rbf
#################################
    

#################################
#   函数：创建多个RBF代理模型    #
#################################
def Creates(DB):
    surr_rbfs = []
    for i in range(0,DB.len_f):
        new_rbf = Create(DB,i)
        surr_rbfs.append(new_rbf)
        
    return surr_rbfs
#################################
    

#################################
#     函数：RBF代理模型预测      #
#################################
def Predict_one(surr_rbf,x):
    x = (x - surr_rbf['dn_r'])/(surr_rbf['up_r'] - surr_rbf['dn_r']) - 0.5
    
    dis_euc = Functional_Operator.Euclidean_Dis(surr_rbf['ax'],x)
    
    r = dis_euc
    Phi = r**3
    
    y = numpy.dot(Phi,surr_rbf['alpha'])
    y = (y + 1)*(surr_rbf['ymax'] - surr_rbf['ymin'])/2 + surr_rbf['ymin']
    return y
#################################
    

#################################
#函数：RBF代理模型预测(多个体预测)#
#################################
def Predict(surr_rbf,x):
    if len(x.shape) == 1:
        N = 1
    else:
        N = numpy.size(x,0)
    
    y_predict = numpy.zeros([N])
    for i in range(0,N):
        y_predict[i] = Predict_one(surr_rbf,x)
        
    return y_predict
#################################
    

###########################################
#函数：RBF代理模型预测(多代理模型多个体预测)#
###########################################
def Predicts(surr_rbfs,x):
    N = numpy.size(x,0)
    num_rbf = len(surr_rbfs)
    
    y_predicts = numpy.zeros([N,num_rbf])
    for i in range(0,num_rbf):
        y_predicts[:,i] = Predict(surr_rbfs[i],x[i,:])
        
    return y_predicts
###########################################