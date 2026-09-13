import numpy as np

X = np.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [-1.0, -2.0],
    [-2.0, -1.0]
])

w = np.array([0.8, 0.6])
b = -0.2

# 1. Compute linear score z
z = X @ w + b

# 2. Apply sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 3. Get probability
p = sigmoid(z)

# 4. Convert probability to class
y_pred = (p > 0.5).astype(int)

def predict(X, w, b):
    z = X @ w + b
    p = sigmoid(z)
    return (p > 0.5).astype(int)    
predictions = predict(X, w, b)
print(predictions)