import numpy as np
import matplotlib.pyplot as plt

# Training parameters
learning_rate = 0.1
epochs = 100

# Initialize parameters
W = np.array([[1, 1], [1, 1]])  # Weight matrix
c = np.array([0, -1])  # Bias vector for hidden layer
w = np.array([1, -2])  # Weight vector for output layer
b = -0.5  # Bias for output layer

# Training loop
losses = []
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X)):
        x = X[i]
        label = y[i]
        # Forward pass
        output, loss, z, h = forward_pass(x, label, W, c, w, b)
        total_loss += loss
        # Compute gradients
        dW, dc, dw, db = compute_gradients(x, label, W, c, w, b, z, h, output)
        # Update parameters
        W -= learning_rate * dW
        c -= learning_rate * dc
        w -= learning_rate * dw
        b -= learning_rate * db
    losses.append(total_loss / len(X))
    print(f"Epoch {epoch + 1}: Loss = {total_loss / len(X):.4f}")

# Plot training loss
plt.plot(range(1, epochs + 1), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.show()

# Find optimal adversarial example
def find_adversarial_example(W, c, w, b, X, y):
    min_distance = float('inf')
    adversarial_example = None
    for i in range(len(X)):
        for j in range(len(X)):
            if i != j and (y[i] - 0.5) * (y[j] - 0.5) < 0:
                x1 = X[i]
                x2 = X[j]
                distance = np.linalg.norm(x1 - x2)
                if distance < min_distance:
                    min_distance = distance
                    adversarial_example = x2
    return adversarial_example

adversarial_example = find_adversarial_example(W, c, w, b, X, y)
print(f"Optimal Adversarial Example: {adversarial_example}")