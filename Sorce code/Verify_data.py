import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from scipy.stats import levene, bartlett

# 生成一些非正态分布的数据作为示例
DB = np.loadtxt('F1_Convergence curve.txt', delimiter=' ')
data = DB[:18, :20].reshape(-1)

# 绘制直方图
plt.hist(data, bins=200, density=True, alpha=0.6, color='g')
#参数依次为： 数据集 直方图中箱子的数量 是否被标准化  透明度  颜色  边缘颜色
plt.title('Histogram of Data')
plt.show()

# 绘制Q-Q图
stats.probplot(data, dist="norm", plot=plt)  #数据集   指定比较（norm 正态， expon 指数， weibull 威布尔）
plt.title('Q-Q Plot')
plt.show()

# 横坐标（Theoretical Quantiles）：
# 横坐标代表理论分位数，即如果数据完全遵循某个理论分布（如正态分布），这些分位数就是该分布下相应概率值所对应的值。
# 例如，在正态分布的Q-Q图中，横坐标的值是标准正态分布的分位数。
# 纵坐标（Sample Quantiles）：
# 纵坐标代表样本分位数，即实际数据集中相应概率值所对应的值。
# 对于数据集中的每个数据点，根据其在排序后的数据集中的位置，计算出对应的分位数，并在图中表示。

# 直线：理想情况下，如果样本数据完全遵循指定的理论分布，那么Q-Q图上的点应该大致沿着一条直线排列。这条直线通常是45度的，表示样本分位数等于理论分位数。
# 偏离直线：如果点显著偏离这条直线，这可能表明样本数据不遵循指定的分布。例如，如果点在直线上方系统地偏离，可能表明样本数据的尾部比理论分布的尾部更重。
# 曲线：如果数据点沿着曲线排列，这可能表明数据存在某种非线性关系或变换。

# 进行Shapiro-Wilk检验
shapiro_test = stats.shapiro(data)
# 返回检验统计量（W），W 值越接近 1，表明样本数据越接近正态分布。如果 W 值远离 1，那么数据越偏离正态分布。
# 如果 p 值小于显著性水平（通常选择 0.05），则拒绝原假设，即认为数据不符合正态分布。
# 如果 p 值大于显著性水平，则不能拒绝原假设，即认为数据可能来自正态分布。
# Shapiro-Wilk 检验通常适用于样本量较小的情况（n < 50），尤其是当数据量在 3 到 5,000 之间时。
# 对于较大的样本量，由于中心极限定理的作用，W 检验统计量是准确的，但 p 值可能不准确
print("Shapiro-Wilk Test:", shapiro_test)

# 进行Kolmogorov-Smirnov检验
ks_test = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
# 数据集  比较类型   用于分布比较的参数，np.mean(data)：这是数据集的样本均值，用作正态分布的均值参数（μ）
#                                 np.std(data)：这是数据集的样本标准差，用作正态分布的标准差参数（σ）
# 返回统计量（D）：D值越大，表明样本分布和参考分布之间的差异越大
# p值：如果p值小于选择的显著性水平（例如0.05），则拒绝原假设，认为样本数据不遵循指定的分布
# K-S检验是一种非参数检验，适用于任何分布的比较，尤其是当样本量较大时，其效果较好。
# 不过，需要注意的是，K-S检验对样本量较大的数据集非常敏感，即使很小的偏差也可能导致拒绝原假设。
print("Kolmogorov-Smirnov Test:", ks_test)

# 使用statsmodels进行正态性检验
sm.qqplot(data, line='s', ax=plt.gca())
# line：
# 指定Q-Q图中的参考线类型。
# 's'：表示添加一条通过样本分位数的标准化直线，这条线考虑了样本均值和标准差，用于比较样本数据与理论分布的一致性。
# '45'：表示添加一条45度参考线，这条线表示完美的一致性，即样本分位数与理论分位数完全匹配。
# 'r'：表示添加一条最佳拟合回归线。
# 'q'：表示通过样本分位数的直线，但不进行标准化。
# None：默认不添加任何参考线。

# plt.gca()：表示获取当前的matplotlib轴对象（AxesSubplot），在这个轴上绘制Q-Q图。
# 如果提供了ax参数，则会在指定的轴上绘制图形；如果没有提供，则会创建一个新的图形。
plt.title('Q-Q Plot (statsmodels)')
plt.show()