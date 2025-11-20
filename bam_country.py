import numpy as np

# ---------------------------------------------------------
# ONE-HOT BIPOLAR ENCODING
# ---------------------------------------------------------
def bipolar_one_hot(index, size):
    """Return a bipolar one-hot vector of given size."""
    vec = -1 * np.ones(size)
    vec[index] = 1
    return vec


# ---------------------------------------------------------
# DEFINE COUNTRY–CAPITAL DATA
# ---------------------------------------------------------

countries = ["India", "France", "Japan", "Italy"]
capitals  = ["New Delhi", "Paris", "Tokyo", "Rome"]

n = len(countries)

# Create bipolar one-hot encodings
X_inputs = np.array([bipolar_one_hot(i, n) for i in range(n)])   # country vectors
Y_targets = np.array([bipolar_one_hot(i, n) for i in range(n)])  # capital vectors

# ---------------------------------------------------------
# BUILD BAM WEIGHT MATRIX
# ---------------------------------------------------------

W = np.zeros((n, n))

for x, y in zip(X_inputs, Y_targets):
    W += np.outer(x, y)    # x^T y

print("\n=== BAM Weight Matrix ===")
print(W)

# ---------------------------------------------------------
# FORWARD AND BACKWARD PASS
# ---------------------------------------------------------

def forward(x, W):
    net = x @ W
    out = np.where(net >= 0, 1, -1)
    return net, out

def backward(y, W):
    net = y @ W.T
    out = np.where(net >= 0, 1, -1)
    return net, out

# ---------------------------------------------------------
# TEST RECALL
# ---------------------------------------------------------

print("\n===============================")
print("  COUNTRY  →  CAPITAL RECALL")
print("===============================\n")

for i, country in enumerate(countries):
    net, out = forward(X_inputs[i], W)
    cap_index = np.argmax(out)   # location of 1 in one-hot encoding
    print(f"{country}  →  {capitals[cap_index]}")

print("\n===============================")
print("  CAPITAL  →  COUNTRY RECALL")
print("===============================\n")

for i, capital in enumerate(capitals):
    net, out = backward(Y_targets[i], W)
    country_index = np.argmax(out)
    print(f"{capital}  →  {countries[country_index]}")
