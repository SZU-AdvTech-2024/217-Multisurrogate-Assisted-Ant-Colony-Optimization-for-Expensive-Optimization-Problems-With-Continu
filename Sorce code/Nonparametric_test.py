from scipy import stats
import numpy as np


def Wilcoxon_Signed_Rank_Test(x,y):

    #用于比较两个配对样本的中位数是否存在显著差异。
    result = stats.wilcoxon(x, y)

    #statistic：检验统计量，即两个样本中位数差异的秩和。
    #pvalue：与检验统计量相关的p值，用于判断统计显著性。
    print("Statistic:", result.statistic)
    print("P-value:", result.pvalue)

def Mann_Whitney_U_Test(x,y):

    #用于比较两个独立样本的中位数是否存在显著差异。
    result = stats.mannwhitneyu(x, y)

    #Ustatistic：检验统计量，即两个样本秩和的差异。U统计量的值越极端（即越大或越小），越有可能拒绝零假设，即两个样本来自不同的分布。
    #pvalue：与检验统计量相关的p值    通常，如果p值小于选择的显著性水平（例如0.05），我们就拒绝零假设，认为两个样本来自不同的分布。
    print("U-statistic:", result.statistic)
    print("P-value:", result.pvalue)

def Kruskal_Wallis_H_Test(x,y,z):

    #用于比较多个独立样本的中位数是否存在显著差异。
    result = stats.kruskal(x, y, z)

    #statistic：检验统计量，即各组秩和与总秩和的比值。
    #pvalue：与检验统计量相关的p值。

    print("H-statistic:", result.statistic)
    print("P-value:", result.pvalue)

if __name__ == "__main__":

    DB1 = np.zeros((21, 600))
    DB1 = np.loadtxt('F1_Convergence curve.txt', delimiter=' ')

    DB2 = np.zeros((21, 600))
    DB2 = np.loadtxt('F1_Convergence curve2.txt', delimiter=' ')

    x = DB1[:18, :20].reshape(-1)
    y = DB2[:18, :20].reshape(-1)

    Wilcoxon_Signed_Rank_Test(x,y)