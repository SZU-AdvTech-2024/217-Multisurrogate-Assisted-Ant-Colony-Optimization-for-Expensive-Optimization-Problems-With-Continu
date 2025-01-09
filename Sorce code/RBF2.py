# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 21:15:22 2019

@author: Katyusha
"""


# from EA import Functional_Operator
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
#################################
#     函数：创建RBF代理模型       #
#################################
import numpy as np

import numpy
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# # 创建 RBF 代理模型函数，结合了 VDM 计算方式
# def Create_VDM(DB, d_r, d_c, dn_r, up_r, t=0, kernel='cubic', Csize=5):
#     ay = DB[1]
#     ax = DB[0]
#
#     ax_r = (ax[:, 0:d_r] - dn_r) / (up_r - dn_r) - 0.5  # 归一化并偏移
#     ax_c = ax  # 类别特征
#
#     ymin = np.min(ay)
#     ymax = np.max(ay)
#     ay = 2 * (ay - ymin) / (ymax - ymin) - 1  # 目标变量归一化
#
#     # 聚类：使用 KMeans 将数据分成 Csize 个聚类
#     km = KMeans(n_clusters=Csize).fit(ax_r)
#     pop_layer = km.labels_
#
#     # 计算每个点到所有其他点的频率分布（基于聚类标签）
#     db_size = len(ay)
#     dis_vdm = np.zeros([db_size, db_size])
#
#     # 计算每个数据点与其他数据点的 VDM 距离
#     for i in range(db_size):
#         dis_vdm[i, :] = calculate_vdm(ax_r, ax_r[i, :], pop_layer, Csize)
#
#     # 计算距离矩阵的核函数
#     r = np.sqrt(dis_vdm ** 2)
#     if kernel == 'gaussian':
#         Phi = (1 / np.sqrt(2 * np.pi)) * np.exp(-(r ** 2) / 1)
#     elif kernel == 'cubic':
#         Phi = r ** 3
#
#     PPhi = np.dot(Phi.T, Phi)
#     PPhinv = np.linalg.pinv(PPhi)
#     Phiy = np.dot(Phi.T, ay)
#     theta = np.dot(PPhinv, Phiy)  # 权重
#
#     # 将 pop_layer 添加到字典中
#     surr_rbf = {
#         "alpha": theta,
#         "ymin": ymin,
#         "ymax": ymax,
#         "ax_r": ax_r,
#         "ax_c": ax_c,
#         "up_r": up_r,
#         "dn_r": dn_r,
#         "kernel": kernel,
#         "Phi": Phi,
#         "pop_layer": pop_layer,  # 添加 pop_layer
#         "Csize": Csize            # 添加 Csize
#     }
#
#     return surr_rbf
#
# # 计算基于频率的 VDM 距离
# def calculate_vdm(ax_r, x_r, pop_layer, Csize):
#     dist = 0.0  # 初始距离为 0
#
#     # 将 ax_r 和 x_r 都转换为一维数组进行对比
#     x_r_layer = pop_layer[np.argmin(np.abs(ax_r - x_r), axis=1)]  # 获取 x_r 所在层级
#     ax_r_layer = pop_layer  # 获取 ax_r 中的聚类层级
#
#     # 确保 layer 数据是整数类型，避免 bincount 报错
#     x_r_layer = x_r_layer.astype(int)
#     ax_r_layer = ax_r_layer.astype(int)
#
#     # 计算频率分布
#     freq_xr = np.bincount(x_r_layer, minlength=Csize) / len(x_r_layer)  # 频率分布
#     freq_axr = np.bincount(ax_r_layer, minlength=Csize) / len(ax_r_layer)
#
#     # 计算频率差异，返回标量（可以通过求和来聚合）
#     dist = np.sum((freq_xr - freq_axr) ** 2)  # 返回标量：频率差异的平方和
#
#     return dist  # 返回的是标量
#
# # 预测单个 RBF 代理模型（基于 VDM 距离）
# def Predict_one_VDM(surr_rbf, x_r):
#     # 确保 x_r 已归一化
#     x_r = (x_r - surr_rbf['dn_r']) / (surr_rbf['up_r'] - surr_rbf['dn_r']) - 0.5
#     x_r = x_r.reshape(1, -1)
#
#     # 调用 calculate_vdm 函数时获取 pop_layer 和 Csize
#     dis_vdm = calculate_vdm(surr_rbf['ax_r'], x_r, surr_rbf['pop_layer'], surr_rbf['Csize'])
#
#     # 继续处理预测逻辑
#     r = np.sqrt(dis_vdm ** 2)
#     if surr_rbf['kernel'] == 'gaussian':
#         Phi = (1 / np.sqrt(2 * np.pi)) * np.exp(-(r ** 2) / 1)
#     elif surr_rbf['kernel'] == 'cubic':
#         Phi = r ** 3
#
#     y = np.dot(Phi, surr_rbf['alpha'])
#     y = (y + 1) * (surr_rbf['ymax'] - surr_rbf['ymin']) / 2 + surr_rbf['ymin']
#     return y
#
# # 预测多个 RBF 代理模型
# def Predict_VDM(surr_rbfs, x_r, x_c):
#     N = np.size(x_r, 0)
#     y_predict = np.zeros([N])
#     for i in range(0, N):
#         y_predict[i] = Predict_one_VDM(surr_rbfs, x_r[i:i + 1, :], x_c[i:i + 1, :])
#
#     return y_predict
#
#
# def Predicts_VDM(surr_rbfs, x_r, x_c):
#     N = np.size(x_r, 0)
#     num_rbf = len(surr_rbfs)
#
#     y_predicts = np.zeros([N, num_rbf])
#     for i in range(0, num_rbf):
#         y_predicts[:, i] = Predict_VDM(surr_rbfs[i], x_r, x_c)
#
#     return y_predicts

#第二次尝试
# def calculate_samp_dist(v_dv):
#     """
#     计算分类变量的采样距离字典 samp_dist
#     :param v_dv: 离散变量的取值。
#     :return: samp_dist - 离散变量的采样距离字典。
#     """
#     l1 = len(v_dv)  # 离散变量的数量
#     samp_dist = {}
#
#     for i in range(l1):
#         n = len(v_dv[i])  # 每个离散变量的取值数目
#         samp_dist[i] = {}
#
#         # 对于每一对离散变量的取值计算距离
#         for i2 in range(n):
#             for i3 in range(i2, n):
#                 ca_value = abs(v_dv[i, i2] - v_dv[i, i3])
#                 samp_dist[i][ca_value] = {}
#                 if i2 == i3:
#                     samp_dist[i][ca_value] = 0  # 如果是相同的取值，距离为 0
#                 else:
#                     # 计算分类变量的加权距离（假设这里是简单的距离，如果需要其他加权计算，可以修改）
#                     samp_dist[i][ca_value] = 1  # 可以根据需要修改为具体的距离计算方式
#
#         # 记录该离散变量的最大值，便于后续归一化
#         samp_dist[i]['max_value'] = max(samp_dist[i].values())
#         if samp_dist[i]['max_value'] == 0:
#             print(samp_dist[i].values())
#
#     return samp_dist
#
#
# def VDM(data_x, v_dv, cxmin, cxmax, r, o):
#     """
#     计算每个样本之间的距离矩阵 distMat，包含连续变量和分类变量的加权距离。
#
#     :param data_x: 输入的数据集，形状为 (n_samples, dim)，其中 dim 是特征的维度。
#     :param v_dv: 离散变量的取值。
#     :param cxmin: 连续变量的最小值，用于归一化。
#     :param cxmax: 连续变量的最大值，用于归一化。
#     :param r: 连续变量的数量（前 r 列为连续变量）。
#     :param o: 离散变量的数量（后 o 列为离散变量）。
#
#     :return: distMat，形状为 (n_samples, n_samples)，每个元素表示样本之间的距离。
#     """
#
#     # 获取样本数量
#     n1 = data_x.shape[0]
#
#     # 初始化距离矩阵
#     rdistMat = np.zeros((n1, n1))
#     cdisMat = np.zeros((n1, n1))
#
#     # 计算连续变量的距离（街区距离）
#     xr = data_x[:, :r]  # 提取连续变量部分
#     for i in range(n1):
#         for j in range(n1):
#             rdistMat[i, j] = np.sum(np.abs(xr[i] - xr[j]) / (cxmax - cxmin))
#
#     # 计算分类变量的距离
#     xr_ca = data_x[:, r:]  # 提取分类变量部分
#
#     # 计算分类变量距离所需的采样距离
#     samp_dist = calculate_samp_dist(v_dv)  # 获取分类变量的距离字典
#
#     for i in range(n1):
#         for j in range(n1):
#             distance = 0
#             for k in range(o):
#                 ca_value = abs(xr_ca[i, k] - xr_ca[j, k])
#                 if samp_dist[k]['max_value'] != 0:
#                     distance += samp_dist[k][ca_value] / samp_dist[k]['max_value']
#             cdisMat[i, j] = distance
#
#     # 归一化距离矩阵
#     D = data_x.shape[1]  # 总的特征维度
#     distMat = (rdistMat + cdisMat) / D
#
#     return distMat



def Create(DB, d_r, d_c, dn_r, up_r, t=0, kernel='cubic'):


    ay = DB[1]
    ax = DB[0]

    ax_r = (ax[:, 0:d_r] - dn_r) / (up_r - dn_r) - 0.5          #直接归一化，再偏移  （-0.5，0.5）
    # ax_r = (DB.x_r - DB.dn_r)/(DB.up_r - DB.dn_r) - 0.5
    ax_c = ax
    # ay = DB.f[:,t]
    ymin = numpy.min(ay)
    ymax = numpy.max(ay)
    ay = 2 * (ay - ymin) / (ymax - ymin) - 1          #  归一化，（-1，1）
    db_size = len(ay)
    dis_euc = numpy.zeros([db_size, db_size])
    dis_ham = numpy.zeros([db_size, db_size])
    for i in range(0, db_size):
        dis_euc[i, :] = Euclidean_Dis1(ax_r, ax_r[i, :], d_r)               #算出i个解与其他解的距离
        # dis_euc[i,:] = Functional_Operator.Euclidean_Dis(ax_r,ax_r[i,:])
        # dis_ham[i,:] = Functional_Operator.Hamming_Dis(ax_c,ax_c[i,:])

    r = numpy.sqrt(dis_euc ** 2)
    if kernel == 'gaussian':
        Phi = (1 / numpy.sqrt(2 * numpy.pi)) * numpy.exp(-(r ** 2) / 1)
    elif kernel == 'cubic':
        Phi = r ** 3

    PPhi = numpy.dot(Phi.T, Phi)
    PPhinv = numpy.linalg.pinv(PPhi)
    Phiy = numpy.dot(Phi.T, ay)
    theta = numpy.dot(PPhinv, Phiy)  # 即权重

    surr_rbf = {"alpha": theta}
    surr_rbf.update({"ymin": ymin})
    surr_rbf.update({"ymax": ymax})
    surr_rbf.update({"ax_r": ax_r})
    surr_rbf.update({"ax_c": ax_c})
    surr_rbf.update({"up_r": up_r})
    surr_rbf.update({"dn_r": dn_r})
    surr_rbf.update({"kernel": kernel})
    surr_rbf.update({"Phi": Phi})
    return surr_rbf              #返回字典


#################################


#################################
#   函数：创建多个RBF代理模型    #
#################################
def Creates(DB, kernel='gaussian'):
    surr_rbfs = []
    for i in range(0, DB.len_f):
        new_rbf = Create(DB, i, kernel)
        surr_rbfs.append(new_rbf)

    return surr_rbfs


#################################


#################################
#     函数：RBF代理模型预测      #
#################################
def Predict_one(surr_rbf, x_r):
    x_r = (x_r - surr_rbf['dn_r']) / (surr_rbf['up_r'] - surr_rbf['dn_r']) - 0.5
    x_r = x_r.reshape(1, -1)
    # dis_euc = Euclidean_Dis(surr_rbf['ax_r'], x_r)
    # dist2 = np.zeros((np.size(x_r, np.size(surr_rbf['ax_r'], 0))))

    # for i in range(np.size(x_r, 0)):
    #     for j in range(np.size(surr_rbf['ax_r'], 0)):
    #         dist2[i] = numpy.linalg.norm(x_r[i, :]- surr_rbf['ax_r'][j, :])

    dis_euc = cdist(x_r, surr_rbf['ax_r'], metric='euclidean')
    # dis_ham = Hamming_Dis2(surr_rbf['ax_c'], x_c, d_c)

    r = numpy.sqrt(dis_euc ** 2)
    if surr_rbf['kernel'] == 'gaussian':
        Phi = (1 / numpy.sqrt(2 * numpy.pi)) * numpy.exp(-(r ** 2) / 1)
    elif surr_rbf['kernel'] == 'cubic':
        Phi = r ** 3

    y = numpy.dot(Phi, surr_rbf['alpha'])
    y = (y + 1) * (surr_rbf['ymax'] - surr_rbf['ymin']) / 2 + surr_rbf['ymin']
    return y


#################################


#################################
# 函数：RBF代理模型预测(多个体预测) #
#################################
def Predict(surr_rbf, x_r, x_c):
    N = numpy.size(x_r, 0)

    y_predict = numpy.zeros([N])
    for i in range(0, N):
        y_predict[i] = Predict_one(surr_rbf, x_r[i:i + 1, :], x_c[i:i + 1, :])

    return y_predict


#################################


###########################################
# 函数：RBF代理模型预测(多代理模型多个体预测) #
###########################################
def Predicts(surr_rbfs, x_r, x_c):
    N = numpy.size(x_r, 0)
    num_rbf = len(surr_rbfs)

    y_predicts = numpy.zeros([N, num_rbf])
    for i in range(0, num_rbf):
        y_predicts[:, i] = Predict(surr_rbfs[i], x_r, x_c)

    return y_predicts


###########################################

def Euclidean_Dis1(x1, x2, d_r):
    l1 = np.size(x1, 0)
    l2 = np.size(x2, 0)
    # dist2 = np.zeros((l1))

    x2 = x2.reshape(int(l2 / d_r), d_r)                 #因为x2是解中的其中一个，所以将x2变形成和x1一样，x2会比x1少一个维度
    dist = cdist(x1, x2, metric='euclidean')            #调用cidst，计算欧式距离          cidst还可以计算曼哈顿，余弦相似度，汉明距离
    # for i in range(l1):
    #     # for j in range(l2):
    #         dist2[i] = numpy.linalg.norm(x1[i, :]-x2)

    return dist.reshape(l1)




