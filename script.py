import numpy as np

def normalize_columns(X):
    # 计算每一列的和
    col_sum = np.sum(X, axis=0)
    # 为避免除以 0，可以在除法前检查或加一个小常数 epsilon
    epsilon = 1e-10
    return X / (col_sum + epsilon)

# 示例：生成一个 3x4 的随机矩阵
X = np.random.rand(3, 4)
Y = normalize_columns(X)

print("随机矩阵 X:")
print(X)
print("\n归一化后的矩阵 Y (每个元素除以所在列的和):")
print(Y)