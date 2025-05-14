import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import Callback

class ErrorTracker(Callback):
    def __init__(self, x_train):
        super().__init__()
        self.x_train = x_train
        self.prev_patterns = None
        self.hamming_distances = []
        self.activation_model = None

    def on_train_begin(self, logs=None):
        # Initialize the activation model after the main model has been built
        layer_outputs = []
        for layer in self.model.layers[1:-1]:
            if isinstance(layer, Dense):
                layer_outputs.append(layer.output)

        self.activation_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=layer_outputs
        )

    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 0:  # Skip first epoch as we need previous patterns
            self.prev_patterns = self.get_activation_patterns(self.x_train)

    def on_epoch_end(self, epoch, logs=None):
        # Calculate errors
        x_test = np.linspace(-2, 2, 1000)
        y_pred = self.model.predict(x_test.reshape(-1, 1), verbose=0)
        y_true = np.sin(np.pi * x_test / 2)

        avg_error = np.mean(np.abs(y_pred.flatten() - y_true))
        max_error = np.max(np.abs(y_pred.flatten() - y_true))

        print(f"Epoch {epoch + 1}: Loss={logs['loss']:.4f}, Avg Error={avg_error:.4f}, Max Error={max_error:.4f}")

        # Calculate Hamming distance if we have previous patterns
        if self.prev_patterns is not None:
            current_patterns = self.get_activation_patterns(self.x_train)
            hamming_dist = np.sum(current_patterns != self.prev_patterns)
            self.hamming_distances.append(hamming_dist)

    def get_activation_patterns(self, x):
        if self.activation_model is None:
            return np.array([])

        # Get activations
        activations = self.activation_model.predict(x, verbose=0)
        if not isinstance(activations, list):
            activations = [activations]

        # Convert to binary patterns (1 for positive, 0 for zero or negative)
        patterns = []
        for activation in activations:
            patterns.append((activation > 0).astype(int))

        # Concatenate all patterns
        if patterns:
            return np.concatenate(patterns, axis=1)
        return np.array([])

# Generate training data
np.random.seed(42)
x_train = np.random.uniform(-2, 2, 200)
y_train = np.sin(np.pi * x_train / 2)

# Create and compile model
model = Sequential([
    Dense(8, activation='relu', input_shape=(1,)),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse')

# Create callback
error_tracker = ErrorTracker(x_train.reshape(-1, 1))

# Train model
history = model.fit(
    x_train.reshape(-1, 1), y_train,
    epochs=100,
    batch_size=32,
    callbacks=[error_tracker],
    verbose=0
)

# Plot training results
plt.figure(figsize=(15, 5))

# Plot 1: Training Error
plt.subplot(131)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot([np.mean(np.abs(model.predict(x_train.reshape(-1, 1), verbose=0).flatten() - y_train)) for _ in range(len(history.history['loss']))],
         label='Avg Error')
plt.plot([np.max(np.abs(model.predict(x_train.reshape(-1, 1), verbose=0).flatten() - y_train)) for _ in range(len(history.history['loss']))],
         label='Max Error')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.title('Training Error')

# Plot 2: Activation Regions
plt.subplot(132)
x_test = np.linspace(-2, 2, 1000).reshape(-1, 1)
patterns = error_tracker.get_activation_patterns(x_test)
unique_patterns = np.unique(patterns, axis=0)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_patterns)))
for i, pattern in enumerate(unique_patterns):
    mask = np.all(patterns == pattern, axis=1)
    plt.fill_between(x_test[mask].flatten(), -1, 1,
                    color=colors[i], alpha=0.3,
                    label=f'Pattern {i+1}')
plt.xlim(-2, 2)
plt.ylim(-1, 1)
plt.title('Activation Regions')
plt.xlabel('Input')

# Plot 3: Hamming Distance
plt.subplot(133)
plt.plot(error_tracker.hamming_distances)
plt.xlabel('Epoch')
plt.ylabel('Hamming Distance')
plt.title('Hamming Distance over Training')

plt.tight_layout()
plt.show()