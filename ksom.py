import numpy as np

# ---------------------------------------------------------
# Input Vectors (4-dimensional)
# ---------------------------------------------------------
inputs = np.array([
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 1]
])

num_inputs = inputs.shape[0]
dim = inputs.shape[1]

# ---------------------------------------------------------
# SOM Parameters
# ---------------------------------------------------------
num_neurons = 2
learning_rate = 0.5
decay = 0.9   # multiplicative LR decay

# ---------------------------------------------------------
# Initialize weights randomly
# ---------------------------------------------------------
weights = np.random.rand(num_neurons, dim)

print("\nInitial Weights:")
print(weights)

# ---------------------------------------------------------
# Explicit Euclidean Distance Function
# ---------------------------------------------------------
def euclidean_distance(x, w):
    return np.sqrt(
        (x[0] - w[0])**2 +
        (x[1] - w[1])**2 +
        (x[2] - w[2])**2 +
        (x[3] - w[3])**2
    )

# ---------------------------------------------------------
# Train SOM
# ---------------------------------------------------------
epochs = 5

for epoch in range(epochs):
    print(f"\n=== Epoch {epoch+1} ===")
    
    for x in inputs:

        # Compute distances manually for each neuron
        distances = []
        for w in weights:
            distances.append(euclidean_distance(x, w))
        distances = np.array(distances)

        # Winner neuron (BMU)
        winner = np.argmin(distances)

        # Update the BMU weights
        weights[winner] = weights[winner] + learning_rate * (x - weights[winner])

        print(f"Input {x} → Winner: Neuron {winner}")
        print("Updated Weights:\n", weights)

    # Reduce learning rate (multiplicative decay)
    learning_rate *= decay
    print(f"New Learning Rate = {learning_rate}")

# ---------------------------------------------------------
# Final Clusters
# ---------------------------------------------------------
print("\n=== Final Clusters ===\n")

for x in inputs:
    distances = []
    for w in weights:
        distances.append(euclidean_distance(x, w))
    distances = np.array(distances)

    winner = np.argmin(distances)
    print(f"Vector {x} belongs to Cluster {winner}")

print("\nFinal SOM Weights:")
print(weights)
