import numpy as np

# ---------------------------------------------------------
# Training Data (From Table 1)
# ---------------------------------------------------------
X = np.array([
    [0, 0, 1, 1],   # Class 1
    [1, 0, 0, 0],   # Class 2
    [0, 0, 0, 1],   # Class 2
    [1, 1, 0, 0],   # Class 1
    [0, 1, 1, 0]    # Class 1
])

y = np.array([1, 2, 2, 1, 1])   # class labels

# ---------------------------------------------------------
# STEP 1: Initialize Prototypes (1 per class)
# Use actual training vectors
# ---------------------------------------------------------
W1 = X[y == 1][0].astype(float)   # first vector of class 1
W2 = X[y == 2][0].astype(float)   # first vector of class 2

weights = {
    1: W1,
    2: W2
}

print("\nInitial prototypes:")
print("W1 (Class 1):", weights[1])
print("W2 (Class 2):", weights[2])

# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------
alpha = 1.0        # initial learning rate (as you requested)
decay = 0.9         # multiplicative decay
epochs = 5          # you can increase if needed

# ---------------------------------------------------------
# Euclidean distance (full formula)
# ---------------------------------------------------------
def dist(x, w):
    return np.sqrt(
        (x[0] - w[0])**2 +
        (x[1] - w[1])**2 +
        (x[2] - w[2])**2 +
        (x[3] - w[3])**2
    )

# ---------------------------------------------------------
# LVQ TRAINING
# ---------------------------------------------------------
for epoch in range(epochs):

    print(f"\n=== Epoch {epoch+1} ===")

    for i in range(len(X)):

        x = X[i]
        cls = y[i]

        # Compute distances to both prototypes
        d1 = dist(x, weights[1])
        d2 = dist(x, weights[2])

        # Winner BMU
        winner = 1 if d1 < d2 else 2

        print(f"\nInput {x}, Class = {cls}")
        print(f"Distances → W1={d1:.3f}, W2={d2:.3f}, Winner = W{winner}")

        # Update rule
        if winner == cls:
            # Correct classification → move closer
            weights[winner] = weights[winner] + alpha * (x - weights[winner])
            print("Correct → Moving prototype TOWARD input")
        else:
            # Wrong classification → move away
            weights[winner] = weights[winner] - alpha * (x - weights[winner])
            print("Wrong → Moving prototype AWAY from input")

        print(f"Updated W{winner} =", weights[winner])

    # Decay learning rate
    alpha *= decay
    print(f"\nNew learning rate = {alpha:.4f}")

# ---------------------------------------------------------
# TESTING
# ---------------------------------------------------------
print("\n=== FINAL TESTING ===")
for i in range(len(X)):
    x = X[i]
    
    d1 = dist(x, weights[1])
    d2 = dist(x, weights[2])
    
    winner = 1 if d1 < d2 else 2
    print(f"Vector {x} → Predicted Class = {winner}")

print("\nFinal Prototypes:")
print("W1:", weights[1])
print("W2:", weights[2])
