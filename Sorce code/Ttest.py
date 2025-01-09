from scipy import stats
import numpy as np


def Ttest_onesamp(DB):

    # 假设我们有一组样本数据
    sample_data = DB[0, :20]
    #  假设的总体均值
    population_mean = DB[0,20]
    # 进行单样本T检验
    t_stat, p_value = stats.ttest_1samp(sample_data, population_mean)
    print("T-statistic:", t_stat)
    print("P-value:", p_value)
    # 如果P值小于显著性水平（例如0.05），则拒绝原假设，即样本均值与总体均值有显著差异
    if p_value < 0.05:
        print("Sample mean is significantly different from the population mean.")
    else:
        print("Sample mean is not significantly different from the population mean.")

def Ttest_doublesamp(DB1, DB2):

    sample_data1 = DB1[0, :20]
    sample_data2 = DB2[0, :20]
    t_stat, p_value = stats.ttest_ind(sample_data1, sample_data2)
    print("T-statistic:", t_stat)
    print("P-value:", p_value)

    # 如果P值小于显著性水平（例如0.05），则拒绝原假设，即两组样本均值有显著差异
    if p_value < 0.05:
        print("Two groups have significantly different means.")
    else:
        print("Two groups do not have significantly different means.")

def Ttest_paired(DB_before, DB_after):
    sample_data_before = DB_before[:18, :20].reshape(-1)
    sample_data_after = DB_after[:18, :20].reshape(-1)
    # 进行配对样本T检验
    t_stat, p_value = stats.ttest_rel(sample_data_before, sample_data_after)

    print("T-statistic:", t_stat)
    print("P-value:", p_value)

    # 如果P值小于显著性水平（例如0.05），则拒绝原假设，即配对样本均值有显著差异
    if p_value < 0.05:
        print("Paired samples have significantly different means.")
    else:
        print("Paired samples do not have significantly different means.")

if __name__ == "__main__":
    DB1 = np.zeros((21, 600))
    DB1 = np.loadtxt('F1_Convergence curve.txt', delimiter=' ')

    DB2 = np.zeros((21, 600))
    DB2 = np.loadtxt('F1_Convergence curve2.txt', delimiter=' ')

    Ttest_paired(DB1, DB2)
