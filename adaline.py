import numpy as np

# ---------------------------------------------------------
# TRAINING DATA (with bias = 1)
# ---------------------------------------------------------
X = np.array([
    [ 1,  1,  1,  1, 1],   # Class +1
    [-1,  1, -1, -1, 1],   # Class +1
    [ 1,  1,  1, -1, 1],   # Class -1
    [ 1, -1, -1,  1, 1]    # Class -1
], dtype=float)

# Targets (t)
t = np.array([1, 1, -1, -1], dtype=float)

# ---------------------------------------------------------
# ADALINE PARAMETERS
# ---------------------------------------------------------
eta = 0.1              # learning rate (stable for ADALINE)
max_epochs = 5
w = np.zeros(X.shape[1], dtype=float)   # initial weights

print("Initial weights:", w)

# ---------------------------------------------------------
# ADALINE TRAINING (LMS Rule)
# ---------------------------------------------------------
for epoch in range(max_epochs):
    sq_error_sum = 0  # for MSE
    
    print(f"\n--- Epoch {epoch+1} ---")
    
    for i in range(len(X)):
        x = X[i]
        target = t[i]
        
        net = np.dot(w, x)       # ADALINE output (linear)
        error = target - net     # LMS error
        
        # LMS weight update
        w = w + eta * error * x
        
        sq_error_sum += error**2
        
        print(f"x={x}, target={target}, net={net:.3f}, error={error:.3f}, updated w={w}")

    mse = sq_error_sum / len(X)
    print("MSE =", mse)

    # stopping condition
    if mse < 0.01:
        print("Converged (MSE < 0.01)")
        break

# ---------------------------------------------------------
# FINAL WEIGHTS
# ---------------------------------------------------------
print("\nFinal weights:", w)

# ---------------------------------------------------------
# TESTING (classification using sign)
# ---------------------------------------------------------
def sign(x):
    return 1 if x >= 0 else -1

print("\n--- FINAL CLASSIFICATION ---")
for i in range(len(X)):
    net = np.dot(w, X[i])
    predicted = sign(net)
    print(f"x={X[i]}, net={net:.2f}, predicted={predicted}, target={t[i]}")
