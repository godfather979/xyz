import numpy as np

# ---------------------------------------------------------
#  STEP 1: INPUT VECTORS (Bipolar)
# ---------------------------------------------------------

E_input = np.array([
    1,  1,  1,
    1, -1,  1,
    1,  1,  1,
    1, -1,  1,
    1,  1,  1
])

F_input = np.array([
    1,  1,  1,
    1, -1,  1,
    1,  1, -1,
    1, -1,  1,
    1, -1,  1
])

# ---------------------------------------------------------
#  STEP 2: TARGET OUTPUTS (Bipolar)
# ---------------------------------------------------------

E_target = np.array([-1, 1])
F_target = np.array([ 1, 1])

# ---------------------------------------------------------
#  STEP 3: WEIGHT MATRIX
# ---------------------------------------------------------

W = np.outer(E_input, E_target) + np.outer(F_input, F_target)

print("\n=== BAM Weight Matrix (15 x 2) ===\n")
print(W)

# ---------------------------------------------------------
#  FORWARD + REVERSE PASS FUNCTIONS
# ---------------------------------------------------------

def forward_net(x, W):
    """Return NET (raw dot product before sign)"""
    return np.dot(x, W)

def forward_pass(x, W):
    y_net = forward_net(x, W)
    y = np.where(y_net >= 0, 1, -1)
    return y_net, y

def reverse_net(y, W):
    """Return NET for reverse pass"""
    return np.dot(y, W.T)

def reverse_pass(y, W):
    x_net = reverse_net(y, W)
    x = np.where(x_net >= 0, 1, -1)
    return x_net, x

# ---------------------------------------------------------
#  TESTING FOR E
# ---------------------------------------------------------

print("\n=============================")
print("   TEST FOR LETTER E")
print("=============================\n")

yE_net, yE = forward_pass(E_input, W)
print("Forward NET (E → y_net):")
print(yE_net)

print("\nForward Output (E → y):")
print(yE)

xE_net, xE = reverse_pass(yE, W)
print("\nReverse NET (y → x_net):")
print(xE_net)

print("\nReverse Output (y → x):")
print(xE)

# ---------------------------------------------------------
#  TESTING FOR F
# ---------------------------------------------------------

print("\n=============================")
print("   TEST FOR LETTER F")
print("=============================\n")

yF_net, yF = forward_pass(F_input, W)
print("Forward NET (F → y_net):")
print(yF_net)

print("\nForward Output (F → y):")
print(yF)

xF_net, xF = reverse_pass(yF, W)
print("\nReverse NET (y → x_net):")
print(xF_net)

print("\nReverse Output (y → x):")
print(xF)
