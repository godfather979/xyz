import numpy as np

# Stored pattern (column vector)
x = np.array([-1, 1, 1, 1]).reshape(4, 1)

# Step 1: Compute weight matrix (NO zero diagonal)
W = x @ x.T
print("Weight Matrix W:\n", W)

# Testing helper
def recall(test):
    test = np.array(test).reshape(4, 1)
    net = W @ test
    out = np.where(net >= 0, 1, -1)  # bipolar sign
    return out.flatten()

# Test cases
tests = {
    "Original": [-1, 1, 1, 1],
    "1 Missing": [0, 1, 1, 1],
    "1 Mistake": [1, 1, 1, 1],
    "2 Missing": [0, 0, 1, 1],
    "2 Mistakes": [1, -1, 1, 1]
}

print("\n--- Testing ---")
for name, vec in tests.items():
    print(f"{name}: input={vec} → output={recall(vec)}")
