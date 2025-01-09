# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 21:15:54 2019

@author: Katyusha
"""

import numpy
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.utils import column_or_1d

#################################
#    函数：创建LSBT代理模型      #
#################################
def Create(DB, d_r, d_c, dn_r, up_r, N_lst, v_dv, t = 0):
#这行定义了一个函数 Create，它接受数据库 DB、实数变量的数量 d_r、类别变量的数量 d_c、
#实数变量的下界 dn_r、实数变量的上界 up_r、每个类别变量的可能值数量列表 N_lst、类别变量的所有可能值 v_dv，以及一个可选参数 t（默认为0）
    ax = DB[0]    #ax赋值DB第一个
    x_c_input = ax[:, d_r:]    #切片得到每行索引d-r之后的数据    应该是得到类变量

    x_c_input2 = np.zeros((np.size(x_c_input,0), d_c))
    # for i in range(np.size(x_c_input,0)):
    #     for j in range(d_c):
    #         for jj in range(N_lst[j]):
    #             if x_c_input[i, j] == v_dv[j,jj]:
    #                 x_c_input[i, j] = jj
    #                 break
                # index = np.argwhere(x_c_input[i, j] == v_dv[j,:])
            # x_c_input2[i, j] = index
    x_r_input = ax[:, 0:d_r]    #切片得到连续变量
    y_input = DB[1]     #DB第二个值为目标函数值
#    y_input = column_or_1d(DB.f[:,t:t+1],warn = True)

    #对类别变量进行One Hot编码
    l_list = []    #用于存储每个类别变量的可能值的范围
    # DB_l = N_lst[1]
    for i in range(d_c):                #类变量的维度
        l_list.append(range(0, N_lst[i]))   #N_lst[i] 类变量所有可能的值
        # 这行代码为每个类别变量创建一个范围，并将其添加到 l_list 中  range 函数：range(start, stop) 函数生成一个从 start 到 stop（不包含 stop）的整数序列
        #range生成从0到N_lst[i]的整数数列        生成了类变量的排序？
    enc = OneHotEncoder(categories = l_list, handle_unknown='ignore')   #将分类变量转换二进制
    enc.fit(x_c_input)                                                     #拟合数据？   .fit 方法用于训练编码器，使其能够了解数据中的类别特征，并为每个类别创建一个唯一的二进制（0/1）表示。  为解中类变量编个码
    x_c_onehot = enc.transform(x_c_input).toarray()                     #将onehot转换成numpy数组
    '''总体onehot用于将类变量编码成唯一二进制形式，.fit用于训练编码器，最后用.transform方法将编码器运用到新数据？'''
    #拼接连续变量和离散变量
    x_input = numpy.hstack((x_r_input, x_c_onehot)) #.hstack用于水平（按列顺序）堆叠数组,将连续变量与onehot编码后的类变量重新组合
#    x_input_class = numpy.hstack((x_r_input,x_c_input))
#    x_input = numpy.hstack((x_r_input,x_c_input))
    
    #训练单个LSBT
    #,criterion="mae"
    surr_lsbt = GradientBoostingRegressor(n_estimators=100,learning_rate=1,max_depth=10,max_leaf_nodes=10, min_samples_split=10, min_samples_leaf=5, random_state=0, loss='absolute_error',tol=0.000001).fit(x_input, y_input)
    '''n_estimators：要使用的树的数量（默认值为 100）。
        learning_rate：学习率，控制每个树对最终预测的贡献（默认值为 0.1）。
        max_depth：每棵树的最大深度（默认值为 3）。更大的深度可能会导致过拟合。
        min_samples_split：拆分内部节点所需的最小样本数（默认值为 2）。
        min_samples_leaf：叶节点所需的最小样本数（默认值为 1）。
        random_state: 这是一个随机数种子，用于确保模型的可重复性。设置为0意味着每次运行代码时，只要输入数据相同，模型的初始化和结果都将是相同的。
        subsample：样本的比例，用于训练每棵树（默认值为 1.0，表示使用所有样本）。  
        loss：损失函数（默认值为 'ls'，即最小二乘法）。 在这里，它被设置为 'absolute_error'，意味着模型将尝试最小化预测值和实际值之间绝对差的平均值。这是一个鲁棒的损失函数，对异常值不太敏感。
        tol: 这是一个容忍度参数，用于早停（early stopping）。如果模型在连续多个提升阶段中性能提升小于这个值，则停止训练。这里设置为 0.000001，意味着如果连续多个提升阶段的性能提升小于 1e-6，则停止训练。
        .fit是训练模型的方法，将x_input, y_input两组数据拿去训练
    '''

#    surr_lsbt = HistGradientBoostingRegressor(max_iter=100,max_depth=10, min_samples_leaf=5,learning_rate=1).fit(x_input_class, DB.f[:,t:t+1])
    surr_lsbt_enc = [surr_lsbt,enc]  #将训练好的lsbt和 onehot编码返回
    return surr_lsbt_enc
#################################
    

#################################
#   函数：创建多个LSBT代理模型    #               没用上的吗？？？？
#################################
def Creates(DB):
    surr_lsbts_enc = []
    for i in range(0,DB.len_f):
        new_lsbt = Create(DB,i)
        surr_lsbts_enc.append(new_lsbt)
        
    return surr_lsbts_enc
#################################
    

#################################
#     函数：LSBT代理模型预测      #
#################################
def Predict(surr_lsbt_enc,x_r,x_c, v_dv):
    #提取模型和 One Hot 编码器
    surr_lsbt = surr_lsbt_enc[0]

    enc = surr_lsbt_enc[1]
    # for i in range(np.size(x_c,0)):
    #     for j in range(np.size(x_c,1)):
    #         index = np.where(x_c[i, j]==v_dv[j,:])
    #         x_c[i, j] = np.array(index).reshape(-1)
    # x_r_input = ax[:, 0:d_r]

    #对离散变量进行One Hot编码

    x_c_onehot = enc.transform(x_c).toarray()
    
    #拼接连续变量和离散变量
    x_input = numpy.hstack((x_r,x_c_onehot))
#    x_input = numpy.hstack((x_r,x_c))

    #采用LSBT预测
    y_predict = surr_lsbt.predict(x_input)
    
    return y_predict  #返回预测值
#################################
    

#################################
#函数：LSBT代理模型预测 (多个模型)#
#################################
def Predicts(surr_lsbts_enc,x_r,x_c):
    N = numpy.size(x_r,0)
    num_rbf = len(surr_lsbts_enc)
    
    y_predicts = numpy.zeros([N,num_rbf])
    for i in range(0,num_rbf):
        y_predicts[:,i] = Predict(surr_lsbts_enc[i],x_r,x_c)
        
    return y_predicts
#################################