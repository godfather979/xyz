import numpy as np

# Training data with bias
X = np.array([
    [ 1,  1,  1,  1, 1],   # Class +1
    [-1,  1, -1, -1, 1],   # Class +1
    [ 1,  1,  1, -1, 1],   # Class -1
    [ 1, -1, -1,  1, 1]    # Class -1
], dtype=float)

# Targets +1 or -1
t = np.array([1, 1, -1, -1], dtype=float)

# Learning rate
eta = 1.0

# Initialize weight vector
w = np.zeros(X.shape[1])

print("Initial Weights:", w)

# Hebbian learning rule
for i in range(len(X)):
    x = X[i]
    target = t[i]
    
    # Hebbian update: w = w + eta * t * x
    w = w + eta * target * x
    
    print(f"\nPattern {i+1}: x={x}, t={target}")
    print(f"Updated weights → {w}")

print("\nFinal Hebbian Weights:", w)

# Testing using sign(w·x)
def sign(net):
    return 1 if net >= 0 else -1

print("\n--- Testing ---")
for i in range(len(X)):
    net = np.dot(w, X[i])
    y = sign(net)
    print(f"x={X[i]}, target={t[i]}, net={net:.2f}, predicted={y}")
