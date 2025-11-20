import numpy as np

# ---------------------------------------------------------
# Training Data (with bias input b = 1)
# ---------------------------------------------------------
X = np.array([
    [ 1,  1,  1,  1, 1],   # belongs → +1
    [-1,  1, -1, -1, 1],   # belongs → +1
    [ 1,  1,  1, -1, 1],   # not belongs → -1
    [ 1, -1, -1,  1, 1]    # not belongs → -1
], dtype=float)

# Target outputs
t = np.array([1, 1, -1, -1], dtype=int)

# ---------------------------------------------------------
# Perceptron parameters
# ---------------------------------------------------------
eta = 1.0     # learning rate
max_epochs = 5
w = np.zeros(X.shape[1], dtype=float)  # initial weights

# ---------------------------------------------------------
# Perceptron activation function (for targets ±1)
# ---------------------------------------------------------
def perceptron_output(net):
    return 1 if net >= 0 else -1

# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------
print("Initial weights:", w)
epoch = 0
converged = False

while epoch < max_epochs and not converged:
    epoch += 1
    any_update = False
    print(f"\n--- Epoch {epoch} ---")

    for i in range(len(X)):
        x = X[i]
        target = t[i]

        net = np.dot(w, x)
        y = perceptron_output(net)

        print(f"Sample {i+1}: x={x}, target={target}, net={net:.3f}, output={y}", end='')

        if y != target:
            # Perceptron weight update: w = w + eta * t * x
            w = w + eta * target * x
            any_update = True
            print(f" → Misclassified → Updated w = {w}")
        else:
            print(" → Correct")

    if not any_update:
        converged = True
        print("\nConverged (no updates this epoch).")

print("\nTraining finished.")
print("Total epochs:", epoch)
print("Final weights (w1, w2, w3, w4, wb):", w)

# ---------------------------------------------------------
# FINAL TESTING
# ---------------------------------------------------------
print("\n--- Final Evaluation ---")
for i in range(len(X)):
    net = np.dot(w, X[i])
    y = perceptron_output(net)
    print(f"x={X[i]}, target={t[i]}, net={net:.2f}, output={y}, {'OK' if y==t[i] else 'WRONG'}")
