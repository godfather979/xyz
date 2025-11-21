import numpy as np

# ---------------------------------------------------------
# Training pairs from your table
# each row: (s1, s2, s3, s4) → (t1, t2)
# ---------------------------------------------------------
S = np.array([
    [1, 0, 1, 0],   # 1st input
    [1, 0, 0, 1],   # 2nd input
    [1, 1, 0, 0],   # 3rd input
    [0, 0, 1, 1]    # 4th input
], dtype=float)

T = np.array([
    [1, 0],   # 1st target
    [1, 0],   # 2nd target
    [0, 1],   # 3rd target
    [0, 1]    # 4th target
], dtype=float)

# ---------------------------------------------------------
# Step 0: Initialize weights (4x2 matrix)
# ---------------------------------------------------------
W = np.zeros((4, 2), dtype=float)
print("Initial Weights:\n", W)

# ---------------------------------------------------------
# Apply Hebbian learning pair-by-pair
# ---------------------------------------------------------
for i in range(len(S)):
    x = S[i].reshape(4, 1)   # column vector
    y = T[i].reshape(1, 2)   # row vector
    
    print(f"\n--- Pair {i+1} ---")
    print("Input x =", S[i])
    print("Target y =", T[i])
    
    # ΔW = x^T * y  (outer product)
    dW = np.dot(x,y)
    print("Weight update ΔW:\n", dW)
    
    # W = W + ΔW
    W = W + dW
    print("Updated Weights:\n", W)

# ---------------------------------------------------------
# Final weight matrix
# ---------------------------------------------------------
print("\n==============================")
print("FINAL WEIGHT MATRIX W:")
print(W)
print("==============================")

# ---------------------------------------------------------
# TEST THE NETWORK
# recall: t_hat = sign(S @ W)
# ---------------------------------------------------------
print("\n--- Testing Recall ---")
for i in range(len(S)):
    net = np.dot(S[i],W)
    out = np.where(net > 0.5, 1, 0)
    print(f"Input {S[i]} → net={net} → predicted={out} → target={T[i]}")
