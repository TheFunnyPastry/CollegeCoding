import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Use a subset for faster training
subset_size = 10000
x_train = x_train[:subset_size]
y_train = y_train[:subset_size]

# Create a CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile with SGD optimizer
optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Custom callback to track gradient norms
class GradientTracker(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.gradient_norms = []
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        # Get a batch of data
        x_batch = self.validation_data[0][:100]
        y_batch = self.validation_data[1][:100]

        # Calculate gradients
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_weights)
            logits = self.model(x_batch, training=False)
            loss = self.model.loss(y_batch, logits)

        # Get gradients
        grads = tape.gradient(loss, self.model.trainable_weights)

        # Calculate gradient norm (Frobenius norm)
        grad_norm = np.sqrt(sum([tf.reduce_sum(tf.square(g)).numpy() for g in grads]))

        # Store metrics
        self.gradient_norms.append(grad_norm)
        self.train_losses.append(logs['loss'])
        self.train_accuracies.append(logs['accuracy'])
        self.val_accuracies.append(logs['val_accuracy'])

        print(f"Epoch {epoch+1}: Gradient Norm = {grad_norm:.4f}")

# Create callback
gradient_tracker = GradientTracker((x_test, y_test))

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=128,
    validation_data=(x_test, y_test),
    callbacks=[gradient_tracker],
    verbose=1
)

# Plot results
plt.figure(figsize=(15, 10))

# Plot 1: Gradient Norms
plt.subplot(2, 2, 1)
plt.plot(gradient_tracker.gradient_norms)
plt.xlabel('Epoch')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norms During Training')

# Plot 2: Loss Function
plt.subplot(2, 2, 2)
plt.plot(gradient_tracker.train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# Plot 3: Training Accuracy
plt.subplot(2, 2, 3)
plt.plot(gradient_tracker.train_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')

# Plot 4: Test Accuracy
plt.subplot(2, 2, 4)
plt.plot(gradient_tracker.val_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')

plt.tight_layout()
plt.savefig('gradient_norm_analysis.png')
plt.show()