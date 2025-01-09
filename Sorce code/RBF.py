# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 21:15:22 2019

@author: Katyusha
"""

import numpy
# from EA import Functional_Operator
from scipy.spatial.distance import cdist

#################################
#     函数：创建RBF代理模型       #
#################################

import numpy as np  # 用于处理数值计算和数组操作
from sklearn.preprocessing import minmax_scale  # 用于数据归一化
from sklearn.cluster import KMeans  # 用于 K-means 聚类
from scipy.spatial.distance import cdist  # 用于计算欧几里得距离（cdist 函数）
from sklearn.preprocessing import minmax_scale

import numpy as np
from scipy.spatial.distance import cdist
from sklearn import preprocessing
from sklearn.cluster import KMeans


#第二次尝试
def calculate_samp_dist(v_dv):
    """
    计算分类变量的采样距离字典 samp_dist
    :param v_dv: 离散变量的取值。
    :return: samp_dist - 离散变量的采样距离字典。
    """
    l1 = len(v_dv)  # 离散变量的数量
    samp_dist = {}

    for i in range(l1):
        n = len(v_dv[i])  # 每个离散变量的取值数目
        samp_dist[i] = {}

        # 对于每一对离散变量的取值计算距离
        for i2 in range(n):
            for i3 in range(i2, n):
                ca_value = abs(v_dv[i, i2] - v_dv[i, i3])
                samp_dist[i][ca_value] = {}
                if i2 == i3:
                    samp_dist[i][ca_value] = 0  # 如果是相同的取值，距离为 0
                else:
                    # 计算分类变量的加权距离（假设这里是简单的距离，如果需要其他加权计算，可以修改）
                    samp_dist[i][ca_value] = 1  # 可以根据需要修改为具体的距离计算方式

        # 记录该离散变量的最大值，便于后续归一化
        samp_dist[i]['max_value'] = max(samp_dist[i].values())
        if samp_dist[i]['max_value'] == 0:
            print(samp_dist[i].values())

    return samp_dist


def VDM(data_x, v_dv, cxmin, cxmax, r, o):
    """
    计算每个样本之间的距离矩阵 distMat，包含连续变量和分类变量的加权距离。

    :param data_x: 输入的数据集，形状为 (n_samples, dim)，其中 dim 是特征的维度。
    :param v_dv: 离散变量的取值。
    :param cxmin: 连续变量的最小值，用于归一化。
    :param cxmax: 连续变量的最大值，用于归一化。
    :param r: 连续变量的数量（前 r 列为连续变量）。
    :param o: 离散变量的数量（后 o 列为离散变量）。

    :return: distMat，形状为 (n_samples, n_samples)，每个元素表示样本之间的距离。
    """

    # 获取样本数量
    n1 = data_x.shape[0]

    # 初始化距离矩阵
    rdistMat = np.zeros((n1, n1))
    cdisMat = np.zeros((n1, n1))

    # 计算连续变量的距离（街区距离）
    xr = data_x[:, :r]  # 提取连续变量部分
    for i in range(n1):
        for j in range(n1):
            rdistMat[i, j] = np.sum(np.abs(xr[i] - xr[j]) / (cxmax - cxmin))

    # 计算分类变量的距离
    xr_ca = data_x[:, r:]  # 提取分类变量部分

    # 计算分类变量距离所需的采样距离
    samp_dist = calculate_samp_dist(v_dv)  # 获取分类变量的距离字典

    for i in range(n1):
        for j in range(n1):
            distance = 0
            for k in range(o):
                ca_value = abs(xr_ca[i, k] - xr_ca[j, k])
                if samp_dist[k]['max_value'] != 0:
                    distance += samp_dist[k][ca_value] / samp_dist[k]['max_value']
            cdisMat[i, j] = distance

    # 归一化距离矩阵
    D = data_x.shape[1]  # 总的特征维度
    distMat = (rdistMat + cdisMat) / D

    return distMat

def Create(v_dv, DB, d_r, d_c, dn_r, up_r, t=0, kernel='gaussian'):
    ay = DB[1]
    ax = DB[0]
    ax_r = ax[:, 0:d_r]  # 连续变量部分
    ax_c = ax[:, d_r:]  # 类别变量部分

    ymin = numpy.min(ay)
    ymax = numpy.max(ay)
    db_size = len(ay)

    # 使用 LVDM 方法计算 r
    # 传入必要的参数，获取样本字典


    r = VDM(ax, v_dv, dn_r, up_r, d_r, d_c)  # 获取 r，从 LVDM 中获取

    # 计算核矩阵 Phi
    if kernel == 'gaussian':
        Phi = (1 / numpy.sqrt(2 * numpy.pi)) * numpy.exp(-(r ** 2) / 1)
    elif kernel == 'cubic':
        Phi = r ** 3

    # 计算回归权重（theta）
    PPhi = numpy.dot(Phi.T, Phi)
    PPhinv = numpy.linalg.pinv(PPhi)
    Phiy = numpy.dot(Phi.T, ay)
    theta = numpy.dot(PPhinv, Phiy)

    # 返回模型结果
    surr_rbf = {"alpha": theta}
    surr_rbf.update({"ymin": ymin})
    surr_rbf.update({"ymax": ymax})
    surr_rbf.update({"ax_r": ax_r})
    surr_rbf.update({"ax_c": ax_c})
    surr_rbf.update({"up_r": up_r})
    surr_rbf.update({"dn_r": dn_r})
    surr_rbf.update({"kernel": kernel})
    surr_rbf.update({"Phi": Phi})

    return surr_rbf

# #第一次尝试
# def data_pro_min_max(input_data):
#     """
#     对输入数据进行归一化，将数据缩放到 [0, 1] 的范围。
#     :param input_data: 输入的原始数据（numpy 数组）
#     :return: 归一化后的数据
#     """
#     norm_data = minmax_scale(input_data)  # 使用 minmax_scale 进行归一化
#     return norm_data
#
#
# def VDM(data_x, data_y, d_r, d_c, v_ca, Csize):
#     """
#     计算每个类别变量值在各个聚类中的分布情况，并计算样本之间的距离 r。
#     :param data_x: 输入数据，包含连续和类别变量
#     :param data_y: 标签（目标值）
#     :param d_r: 连续变量的维度
#     :param d_c: 类别变量的维度
#     :param v_ca: 类别变量的所有可能取值
#     :param Csize: 聚类数量
#     :return: 返回计算得到的 r
#     """
#
#     # 数据拆分为连续和类别变量
#     data_cnx = data_x[:, :d_r]  # 连续变量部分
#     data_cax = data_x[:, d_r:]  # 类别变量部分
#
#     # 归一化连续变量
#     norm_data_cnx = data_pro_min_max(data_cnx)
#
#     # 聚类：计算聚类分布情况
#     km = KMeans(n_clusters=Csize).fit(data_y.reshape(-1, 1))
#     pop_layer = km.labels_  # 聚类标签
#
#     # 计算样本之间的距离 r
#     db_size = len(data_x)
#     dis_euc = np.zeros([db_size, db_size])
#     dis_ham = np.zeros([db_size, db_size])
#
#     for i in range(0, db_size):
#         for j in range(0, db_size):
#             if i != j:
#                 # 连续变量部分距离
#                 d_cnx = np.sum(np.abs(norm_data_cnx[i] - norm_data_cnx[j]))  # 连续变量部分的距离
#                 d_cnx_normalized = d_cnx / (np.max(norm_data_cnx) - np.min(norm_data_cnx))  # 归一化
#                 d_cnx_weighted = d_cnx_normalized  # 直接使用归一化值
#
#                 # 类别变量部分距离
#                 d_cax = np.sum(data_cax[i] != data_cax[j])  # 汉明距离
#                 d_cax_normalized = d_cax / d_c  # 归一化
#                 d_cax_weighted = d_cax_normalized  # 直接使用归一化值
#
#                 # 综合距离（公式中的总距离）
#                 total_distance = (d_cnx_weighted + d_cax_weighted) / (d_r + d_c)  # 归一化后的综合距离
#
#                 # 存储欧几里得和汉明距离
#                 dis_euc[i, j] = d_cnx_weighted
#                 dis_ham[i, j] = d_cax_weighted
#
#     # 综合距离 r
#     r = np.sqrt(dis_euc ** 2 + dis_ham ** 2)  # 计算总的距离
#
#     return r

# def Create(DB, d_r, d_c, dn_r, up_r, t=0, kernel='gaussian'):
#     ay = DB[1]
#     ax = DB[0]
#     ax_r = ax[:, 0:d_r]  # 连续变量部分
#     ax_c = ax[:, d_r:]  # 类别变量部分
#
#     ymin = numpy.min(ay)
#     ymax = numpy.max(ay)
#     db_size = len(ay)
#
#     # 使用 LVDM 方法计算 r
#     # 传入必要的参数，获取样本字典
#     Csize = 100  # 设定聚类数量为 25
#     v_ca = []  # 你需要定义 v_ca，根据你的类变量取值来构造 v_ca
#
#     r = VDM(ax, ay, d_r, d_c, v_ca, Csize)  # 获取 r，从 LVDM 中获取
#
#     # 计算核矩阵 Phi
#     if kernel == 'gaussian':
#         Phi = (1 / numpy.sqrt(2 * numpy.pi)) * numpy.exp(-(r ** 2) / 1)
#     elif kernel == 'cubic':
#         Phi = r ** 3
#
#     # 计算回归权重（theta）
#     PPhi = numpy.dot(Phi.T, Phi)
#     PPhinv = numpy.linalg.pinv(PPhi)
#     Phiy = numpy.dot(Phi.T, ay)
#     theta = numpy.dot(PPhinv, Phiy)
#
#     # 返回模型结果
#     surr_rbf = {"alpha": theta}
#     surr_rbf.update({"ymin": ymin})
#     surr_rbf.update({"ymax": ymax})
#     surr_rbf.update({"ax_r": ax_r})
#     surr_rbf.update({"ax_c": ax_c})
#     surr_rbf.update({"up_r": up_r})
#     surr_rbf.update({"dn_r": dn_r})
#     surr_rbf.update({"kernel": kernel})
#     surr_rbf.update({"Phi": Phi})
#
#     return surr_rbf

# def VDM():
#
#     n1 = data_point.shape[0]
#     n2 = centers.shape[0]
#     D = data_point.shape[1]
#
#     # 连续变量（街区距离）
#     rdistMat = np.zeros((n1, n2))
#     xr1 = data_point[:, :r]
#     xr2 = centers[:, :r]
#     xr1_ca = data_point[:, r:]
#     xr2_ca = centers[:, r:]
#     x12 = np.concatenate((xr1, xr2), axis=0)
#
#
#     cdisMat = np.zeros((n1, n2))
#     for i in range(n1):
#         rdistMat[i, :] = np.sum(np.abs(xr1[i] - xr2) / (cxmax - cxmin),
#                                 axis=1)  ############################修改在这里######################################
#
#         for i2 in range(n2):
#             distance = 0
#             for i3 in range(self.o):
#                 ca_value = abs(xr1_ca[i, i3] - xr2_ca[i2, i3])
#                 if self.samp_dist[i3]['max_value'] != 0:
#                     distance = distance + self.samp_dist[i3][ca_value] / self.samp_dist[i3]['max_value']
#             cdisMat[i, i2] = distance
#
#     distMat = (rdistMat + cdisMat) / D
#     return distMat


# def Create(DB, d_r, d_c, dn_r, up_r, t = 0, kernel = 'gaussian'):
#     ay = DB[1]
#     ax = DB[0]
#     ax_r = ax[:, 0:d_r]
#
#     # ax_r = (DB.x_r - DB.dn_r)/(DB.up_r - DB.dn_r) - 0.5
#     ax_c = ax[:, d_r:]
#     # ay = DB.f[:,t]
#     ymin = numpy.min(ay)
#     ymax = numpy.max(ay)
#     # ay = 2*(ay - ymin)/(ymax - ymin) - 1
#     db_size = len(ay)
#     dis_euc = numpy.zeros([db_size, db_size])
#     dis_ham = numpy.zeros([db_size, db_size])
#     for i in range(0,db_size):
#         dis_euc[i, :] = Euclidean_Dis1(ax_r, ax_r[i,:], d_r, dn_r, up_r)
#         dis_ham[i, :] = Hamming_Dis1(ax_c,ax_c[i,:], d_c)/d_c
#         # dis_euc[i,:] = Functional_Operator.Euclidean_Dis(ax_r,ax_r[i,:])
#         # dis_ham[i,:] = Functional_Operator.Hamming_Dis(ax_c,ax_c[i,:])
#     r = VDM()
#     # r = numpy.sqrt(dis_euc**2 + dis_ham**2)
#     # r = dis_euc + dis_ham
#     if kernel == 'gaussian':
#         Phi = (1/numpy.sqrt(2*numpy.pi) )*numpy.exp( -(r**2)/1 )
#     elif kernel == 'cubic':
#         Phi = r**3
#
#     PPhi = numpy.dot(Phi.T,Phi)
#     PPhinv = numpy.linalg.pinv(PPhi)
#     Phiy = numpy.dot(Phi.T, ay)
#     theta = numpy.dot(PPhinv,Phiy)#即权重
#
#     surr_rbf = {"alpha":theta}
#     surr_rbf.update({"ymin":ymin})
#     surr_rbf.update({"ymax":ymax})
#     surr_rbf.update({"ax_r":ax_r})
#     surr_rbf.update({"ax_c":ax_c})
#     surr_rbf.update({"up_r": up_r})
#     surr_rbf.update({"dn_r": dn_r})
#     surr_rbf.update({"kernel":kernel})
#     surr_rbf.update({"Phi":Phi})
#     return surr_rbf
#################################
    

#################################
#   函数：创建多个RBF代理模型    #
#################################
def Creates(DB,kernel = 'gaussian'):

    surr_rbfs = []
    for i in range(0,DB.len_f):
        new_rbf = Create(DB,i,kernel)
        surr_rbfs.append(new_rbf)
        
    return surr_rbfs
#################################
    

#################################
#     函数：RBF代理模型预测      #
#################################
def Predict_one(surr_rbf, x_r, x_c, d_r, d_c):               # 下午。。。
    # x_r = (x_r - surr_rbf['dn_r'])/(surr_rbf['up_r'] - surr_rbf['dn_r']) - 0.5

    # dis_euc = Euclidean_Dis(surr_rbf['ax_r'], x_r, )
    # dis_euc = cdist(x_r, surr_rbf['ax_r'], metric='euclidean')/d_r

    dis_euc = Euclidean_Dis2(surr_rbf['ax_r'], x_r, d_r, surr_rbf['dn_r'], surr_rbf['up_r'])
    dis_ham = Hamming_Dis2(surr_rbf['ax_c'], x_c, d_c)/d_c        #算两个距离

    r = numpy.sqrt(dis_euc**2 + dis_ham**2)
    # r = dis_euc + dis_ham
    if surr_rbf['kernel'] == 'gaussian':
        Phi = (1/numpy.sqrt(2*numpy.pi) )*numpy.exp( -(r**2)/1 )
    elif surr_rbf['kernel'] == 'cubic':
        Phi = r**3
    
    y = numpy.dot(Phi,surr_rbf['alpha'])            #计算和函数和参数的点积？
    y = (y + 1)*(surr_rbf['ymax'] - surr_rbf['ymin'])/2 + surr_rbf['ymin']            #进行缩放和偏移到y的范围中
    return y
#################################
    

#################################
#函数：RBF代理模型预测(多个体预测) #
#################################
def Predict(surr_rbf,x_r,x_c):
    N = numpy.size(x_r,0)
    
    y_predict = numpy.zeros([N])
    for i in range(0,N):
        y_predict[i] = Predict_one(surr_rbf,x_r[i:i+1,:],x_c[i:i+1,:])
        
    return y_predict
#################################
    

###########################################
#函数：RBF代理模型预测(多代理模型多个体预测) #
###########################################
def Predicts(surr_rbfs,x_r,x_c):
    N = numpy.size(x_r,0)
    num_rbf = len(surr_rbfs)
    
    y_predicts = numpy.zeros([N,num_rbf])
    for i in range(0,num_rbf):
        y_predicts[:,i] = Predict(surr_rbfs[i],x_r,x_c)
        
    return y_predicts
###########################################

def Euclidean_Dis1(x1, x2, d_r, dn_r, up_r):
    l1 = np.size(x1, 0)
    l2 = np.size(x2, 0)
    x2 = x2.reshape(int(l2 / d_r), d_r)

    # x2 = x2.reshape(int(l2/d_r), d_r)
    dist2 = np.zeros((int(l2 / d_r), l1))
    # dist = cdist(x1, x2, metric='cityblock')
    for i in range(l1):
        for j in range(int(l2/d_r)):
            dist2[j, i] = np.sqrt(np.sum(np.abs(x1[i, :] - x2[j, :])/(up_r - dn_r))**2)
    dist2 = dist2/d_r
    return dist2

def Euclidean_Dis2(x1, x2, d_r, dn_r, up_r):
    l1 = np.size(x1, 0)
    l2 = np.size(x2, 0)
    # x2 = x2.reshape(int(l2 / d_r), d_r)

    # x2 = x2.reshape(int(l2/d_r), d_r)
    dist2 = np.zeros((l2, l1))
    # dist = cdist(x1, x2, metric='cityblock')
    for i in range(l1):
        for j in range(l2):
            dist2[j, i] = np.sqrt(np.sum(np.abs(x1[i, :] - x2[j, :])/(up_r - dn_r))**2)         #有做归一化
    dist2 = dist2/d_r               #将数据缩放，同样是归一化
    return dist2


def Hamming_Dis1(x1, x2, d_c):
    l1 = np.size(x1, 0)
    l2 = np.size(x2, 0)
    x2 = x2.reshape(int(l2 / d_c), d_c)

    dist = np.zeros((int(l2 / d_c), l1))
    for i in range(l1):
        for j in range(int(l2 / d_c)):
            dist2 = 0
            for jj in range(d_c):
                if x1[i, jj] != x2[j, jj]:
                    dist2 += 1
                else:
                    dist2 += 0
            dist[j, i] = dist2

    return dist.reshape(l1)


# def Euclidean_Dis2(x1, x2):
#     # l1 = np.size(x1, 0)
#     # l2 = np.size(x2, 0)
#     # # dist2 = np.zeros((l1))
#     #
#     # x2 = x2.reshape(int(l2/d_r), d_r)
#     dist = cdist(x1, x2, metric='euclidean')
#
#     return dist.reshape(l1)


def Hamming_Dis2(x1, x2, d_c):
    l1 = np.size(x1, 0)
    l2 = np.size(x2, 0)
    # x2 = x2.reshape(int(l2 / d_c), d_c)
    #
    dist = np.zeros((l2, l1))
    for i in range(l1):
        for j in range(l2):
            dist2 = 0
            for jj in range(d_c):
                if x1[i, jj] != x2[j, jj]:
                    dist2 += 1
                else:
                    dist2 += 0
            dist[j, i] = dist2

    return dist
