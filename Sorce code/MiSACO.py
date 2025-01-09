# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 17:07:28 2020

@author: Katyusha
"""

import numpy
import datetime
from EA import EA_Class
from EA import EA_Operator
from TestFunction import EOPCC_Uncons
from Surrogate import Kriging
from Surrogate import LSBT
from Surrogate import RBF
from LocalSearch.RBF_LS import RBF_LS
import datetime

#################################
#         算法：MiSACO          #
#################################
def MiSACO(Par):
    # ACO_MV算子参数初始化
    Par_ACO_MV = Par['Par_ACO_MV']
    # 函数评价初始化
    FEs = Par['pop_size']
    # 种群初始化
    Pop = EA_Operator.Initialize(Par)
    DB  = EA_Class.DataBase(Pop,Par)
    while FEs < Par['MaxFEs']:
        DB,Pop,FEs = MS_Selection_Rand(DB,Pop,Par_ACO_MV,Par,FEs)
        DB,Pop,FEs = SALS(DB,Pop,Par,FEs)
            
        if "UI" in Par.keys():
            ui = Par['UI']
            print_flag = FEs%1
            if print_flag == 0:
                ui.print_TB(Pop,FEs)
                print('当前最优值为：%E' %Pop.best_f)
                print('当前连续变量最优解为:')
                print(Pop.best_x_r[0,:])
                print('当前离散变量最优解为:')
                print(Pop.best_x_c[0,:])
                print('当前函数评价为：%d' %FEs)

                value,FEs_list = DB.Converge_curve()
                ui.plot_curve(value,FEs_list)
    return Pop,DB
#################################
    

#################################
#   多模型选择策略(随机选择)     #
#################################
def MS_Selection_Rand(DB,Pop,Par_ACO_MV,Par,FEs):
    ###################################################################
    # 代理模型建立
    ###################################################################
    #建立两种代理模型
    start = datetime.datetime.now()
    surr_lsbt = LSBT.Creates(DB)
    end = datetime.datetime.now()
    print(end-start)

    surr_rbf = RBF.Creates(DB)
    end = datetime.datetime.now()
    print(end-start)
    ###################################################################
    
    
    ###################################################################
    # 代理模型预选择
    ###################################################################
    # 基于LSBT的预选择
    x_r_generate,x_c_generate = EA_Operator.ACO_MV_generates(Pop,Par_ACO_MV)
    f_lsbt = LSBT.Predicts(surr_lsbt,x_r_generate,x_c_generate)
    Off_lsbt = EA_Class.Offspring(x_r_generate, x_c_generate, f_lsbt)
    Off_lsbt = Off_lsbt.GetBestOff()
    
    # 基于RBF的预选择
    x_r_generate,x_c_generate = EA_Operator.ACO_MV_generates(Pop,Par_ACO_MV)
    f_rbf = RBF.Predicts(surr_rbf,x_r_generate,x_c_generate)
    Off_rbf = EA_Class.Offspring(x_r_generate, x_c_generate, f_rbf)
    Off_rbf = Off_rbf.GetBestOff()
    
    # 随机预选择
    x_r_generate,x_c_generate = EA_Operator.ACO_MV_generates(Pop,Par_ACO_MV)
    f_rand = numpy.random.random([Par_ACO_MV['m'],Pop.len_f])
    Off_rand = EA_Class.Offspring(x_r_generate, x_c_generate, f_rand)
    Off_rand = Off_rand.GetBestOff()
    ###################################################################
    
    
    ###################################################################
    # 评估预选择得到的个体
    ###################################################################
    # 评估LSBT选择的后代
    f_lsbt_real = Par['fit_func'](Off_lsbt.x_r,Off_lsbt.x_c)
    Off_lsbt_real = EA_Class.Offspring(Off_lsbt.x_r, Off_lsbt.x_c, f_lsbt_real)
    
    # 评估RBF选择的后代
    f_rbf_real = Par['fit_func'](Off_rbf.x_r,Off_rbf.x_c)
    Off_rbf_real = EA_Class.Offspring(Off_rbf.x_r, Off_rbf.x_c, f_rbf_real)
    
    # 评估随机选择的后代
    f_rand_real = Par['fit_func'](Off_rand.x_r,Off_rand.x_c)
    Off_rand_real = EA_Class.Offspring(Off_rand.x_r, Off_rand.x_c, f_rand_real)
    ###################################################################
    
    
    ###################################################################
    # 更新种群
    ###################################################################     
    # 根据LSBT选择的个体更新种群
    dx = numpy.abs( numpy.hstack((Off_lsbt_real.x_r,Off_lsbt_real.x_c)) - numpy.hstack((DB.x_r,DB.x_c)) )
    meadis = numpy.min( numpy.mean(dx,1) )
    if meadis > 1e-4:
        Pop = Pop.ElitistSelection(Off_lsbt_real)
        DB.Update(Off_lsbt_real)
        FEs = FEs + 1
        
    # 根据RBF选择的个体更新种群
    dx = numpy.abs( numpy.hstack((Off_rbf_real.x_r,Off_rbf_real.x_c)) - numpy.hstack((DB.x_r,DB.x_c)) )
    meadis = numpy.min( numpy.mean(dx,1) )
    if meadis > 1e-4:
        Pop = Pop.ElitistSelection(Off_rbf_real)
        DB.Update(Off_rbf_real)
        FEs = FEs + 1
        
    # 根据随机选择的个体更新种群
    dx = numpy.abs( numpy.hstack((Off_rand_real.x_r,Off_rand_real.x_c)) - numpy.hstack((DB.x_r,DB.x_c)) )
    meadis = numpy.min( numpy.mean(dx,1) )
    if meadis > 1e-4:
        Pop = Pop.ElitistSelection(Off_rand_real)
        DB.Update(Off_rand_real)
        FEs = FEs + 1
    ###################################################################
    return DB, Pop, FEs
#################################
    

#################################
#           局部搜索             #
#################################
def SALS(DB,Pop,Par,FEs):
    Sub_DB = DB.Same_DB(Pop.best_x_c)
    if Sub_DB.db_size>25:
        f_local,x_local = RBF_LS(Sub_DB,Pop.best_x_r)
        f_local = Par['fit_func'](x_local,Pop.best_x_c)
        Off_local = EA_Class.Offspring(x_local, Pop.best_x_c, f_local)
        dx = numpy.abs( numpy.hstack((Off_local.x_r,Off_local.x_c)) - numpy.hstack((DB.x_r,DB.x_c)) )
        meadis = numpy.min( numpy.mean(dx,1) )
        if meadis > 1e-4:
            Pop = Pop.ElitistSelection(Off_local)
            DB.Update(Off_local)
            FEs = FEs + 1
    return DB, Pop, FEs
#################################