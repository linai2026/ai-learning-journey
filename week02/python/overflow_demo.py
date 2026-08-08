import math
import torch
import numpy as np

print(math.exp(10))
print(math.exp(100))
print(math.exp(500))

print(torch.exp(torch.tensor(10.)))
print(torch.exp(torch.tensor(100.)))
print(torch.exp(torch.tensor(500.)))

#下溢
values = np.array([-10, -100, -1000], dtype=np.float64)
print(np.exp(values))

#Softmax数值稳定性
#先实现不稳定版本：
def naive_softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()

x = np.array([1000.0, 1001.0, 1002.0])
# print(naive_softmax(x))  # 输出为nan

#再实现稳定版本
def stable_softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum()

print(stable_softmax(x))  # 输出为[0.09003057 0.24472847 0.66524096]
