import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of sigmoid
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# ReLU activation function
def relu(z):
    return np.maximum(0, z)

# Derivative of ReLU
def relu_derivative(z):
    return (z > 0).astype(float)

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
y = np.array([0, 1, 1, 0])  # Labels

# Initialize parameters
W = np.array([[1, 1], [1, 1]], dtype=float)  # Weight matrix
c = np.array([0, -1], dtype=float)  # Bias vector for hidden layer
w = np.array([1, -2], dtype=float)  # Weight vector for output layer
b = -0.5  # Bias for output layer

# Forward pass and loss computation
def forward_pass(x, y, W, c, w, b):
    z = np.dot(W, x) + c  # Linear transformation
    h = relu(z)  # ReLU activation
    output = sigmoid(np.dot(w, h) + b)  # Sigmoid activation
    loss = - (y * np.log(output) + (1 - y) * np.log(1 - output))  # Cross-entropy loss
    return output, loss, z, h

# Compute gradients
def compute_gradients(x, y, W, c, w, b, z, h, output):
    dL_doutput = output - y  # Gradient of loss w.r.t. output
    doutput_dz2 = sigmoid_derivative(np.dot(w, h) + b)  # Gradient of sigmoid
    dz2_dh = w  # Gradient of linear combination w.r.t. hidden layer
    dh_dz1 = relu_derivative(z)  # Gradient of ReLU
    dz1_dW = np.outer(x, np.ones_like(c))  # Gradient of linear transformation w.r.t. W

    # Gradients for each parameter
    dL_db = dL_doutput * doutput_dz2
    dL_dw = dL_doutput * doutput_dz2 * h
    dL_dh = dL_doutput * doutput_dz2 * dz2_dh
    dL_dz1 = dL_dh * dh_dz1
    dL_dW = np.outer(dL_dz1, x)
    dL_dc = dL_dz1

    return dL_dW, dL_dc, dL_dw, dL_db

# Problem 2: Compute outputs, losses, and gradients for all samples
print("Problem 2: Outputs, Losses, and Gradients")
for i in range(len(X)):
    x = X[i]
    label = y[i]
    output, loss, z, h = forward_pass(x, label, W, c, w, b)
    dW, dc, dw, db = compute_gradients(x, label, W, c, w, b, z, h, output)
    print(f"Sample {i + 1}:")
    print(f"  Input: {x}, Label: {label}")
    print(f"  Output: {output:.4f}, Loss: {loss:.4f}")
    print(f"  Gradients: dW={dW}, dc={dc}, dw={dw}, db={db}")
    print()

#training the network
print("Problem 3: Training the Network")
learning_rate = 0.1
epochs = 100
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