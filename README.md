# Practical 1: Simulate a Perceptron using NumPy

## Aim
To implement a single-layer Perceptron from scratch using NumPy and use it to classify PTAL (Public Transport Accessibility Level) as **Good** or **Poor**, based on walk time to nearest stop and service frequency.

## Theory

**Perceptron** is the simplest form of an artificial neural network, introduced by Frank Rosenblatt (1958). It models a single biological neuron and is used for **binary linear classification** — separating data into two classes using a straight line (or hyperplane in higher dimensions).

**Structure:**
- Inputs: $x_1, x_2, ..., x_n$ (features)
- Weights: $w_1, w_2, ..., w_n$ (importance of each feature)
- Bias: $b$ (shifts the decision boundary)
- Weighted sum: $z = \sum_{i=1}^{n} w_i x_i + b$
- Activation (step function): $\hat{y} = 1 \text{ if } z \geq 0, \text{ else } 0$

**Learning rule (Perceptron Learning Algorithm):**
For each training sample, weights are updated only when the prediction is wrong:
$$w_i \leftarrow w_i + \eta (y - \hat{y}) x_i$$
$$b \leftarrow b + \eta (y - \hat{y})$$
where $\eta$ is the learning rate, $y$ is the true label, and $\hat{y}$ is the predicted label. This is repeated over multiple epochs until the weights converge (error becomes 0 for all samples), guaranteed only if the data is **linearly separable**.

**Applied context — PTAL accessibility:**
PTAL scores an area's public transport access using walk time to stops and service frequency. Here the perceptron takes two normalized features — `walk_time` and `service_freq` — and learns a linear boundary separating **Good access** (short walk, high frequency) from **Poor access** (long walk, low frequency), mimicking how a simplified accessibility classifier would work.

**Limitation:** A single perceptron can only learn linearly separable functions (e.g., AND, OR) — it cannot learn XOR or other non-linearly separable patterns, which is why multi-layer networks (MLPs) were later developed.

# 1. Simulate a Perceptron using NumPy
# Domain: PTAL (Public Transport Accessibility Level) classification
# Given [avg_walk_time_to_stop_min, weighted_service_freq_per_hr], predict
# whether a location has Good (1) or Poor (0) transit accessibility.
import numpy as np

# Step activation function
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptron class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.lr = learning_rate

    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return step_function(z)

    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

# Training data: [walk_time_min (normalized), service_freq_per_hr (normalized)]
# Label 1 = Good PTAL accessibility, 0 = Poor accessibility
# Rule of thumb: short walk time AND high frequency -> good access
X = np.array([
    [0.1, 0.9],   # 2 min walk, high freq -> good
    [0.2, 0.8],   # short walk, high freq -> good
    [0.9, 0.1],   # long walk, low freq -> poor
    [0.8, 0.2],   # long walk, low freq -> poor
    [0.15, 0.85], # good
    [0.85, 0.15]  # poor
])
y = np.array([1, 1, 0, 0, 1, 0])

# Train the perceptron
p = Perceptron(input_size=2)
p.train(X, y, epochs=10)

# Test predictions
labels = {1: 'Good Access', 0: 'Poor Access'}
print("PTAL Accessibility Predictions:")
for xi in X:
    pred = p.predict(xi)
    print(f"Input (walk={xi[0]:.2f}, freq={xi[1]:.2f}) -> {labels[pred]}")
