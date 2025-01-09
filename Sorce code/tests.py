from sklearn.preprocessing import OneHotEncoder
import numpy as np

# 假设我们有以下类别特征数据
categories = [['red', 'blue', 'green'], ['cat', 'dog', 'bird']]

# 创建 OneHotEncoder 实例
enc = OneHotEncoder(categories=categories, handle_unknown='ignore')

# 训练编码器
enc.fit([['red', 'cat'], ['blue', 'dog'], ['green', 'bird']])

# 将编码器应用到新数据上
transformed = enc.transform([['blue', 'bird'], ['red', 'dog']])

# 将稀疏矩阵转换为 NumPy 数组
transformed_array = transformed.toarray()

print(transformed_array)