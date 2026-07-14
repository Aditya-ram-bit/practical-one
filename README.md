# Practical-1
# Perceptron Simulation using NumPy — Explanation

This notebook implements a single-layer perceptron — the simplest kind of artificial neuron — and trains it to learn the AND logic gate. Here's what's happening step by step:

## 1. The Step Activation Function

```python
def step_function(x):
    return 1 if x >= 0 else 0
```

This is the perceptron's "decision maker." Once the neuron computes a weighted sum of inputs, this function converts that number into a binary output: fire (1) if the sum is ≥ 0, stay silent (0) otherwise. This mimics a biological neuron's all-or-nothing firing behavior.

## 2. The Perceptron Class

**Initialization (`__init__`)**
- `weights = np.zeros(input_size)` — starts with zero weight for each input (here, 2 inputs)
- `bias = 0` — starts at zero too
- `lr = 0.1` — the learning rate, controlling how big a step the weights take with each correction

**Prediction (`predict`)**

```python
z = np.dot(x, self.weights) + self.bias
return step_function(z)
```

This computes `z = w1*x1 + w2*x2 + bias`, then passes it through the step function to get 0 or 1.

**Training (`train`)** — this is the heart of the perceptron learning rule:

```python
prediction = self.predict(xi)
error = target - prediction
self.weights += self.lr * error * xi
self.bias += self.lr * error
```

For every training example, the perceptron:

1. Makes a prediction
2. Computes the error (target − prediction)
3. Nudges the weights and bias in the direction that reduces that error

If the prediction is correct, `error = 0`, so nothing changes. If wrong, the weights shift proportionally to the learning rate and the input values. This process repeats for every sample, across `epochs = 10` full passes through the data — giving the perceptron multiple chances to converge on the correct weights.

## 3. The Training Data — AND Gate

```python
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])
```

This is the truth table for logical AND: the output is only 1 when both inputs are 1. The perceptron's job is to find a weight/bias combination that draws a straight line (a linear decision boundary) separating the single "1" case from the three "0" cases — which is possible here because AND is a linearly separable function (this matters because perceptrons cannot learn non-linearly-separable functions like XOR).

## 4. The Output

```
Predictions:
Input: [0 0], Output: 0
Input: [0 1], Output: 0
Input: [1 0], Output: 0
Input: [1 1], Output: 1
```

After 10 epochs of training, the perceptron predicts correctly for all four input combinations — it has successfully learned the AND gate. This confirms:

- The weights/bias converged to values where `w1*x1 + w2*x2 + bias ≥ 0` only when both inputs are 1
- The perceptron learning rule works as expected on a linearly separable problem
- 10 epochs was enough for convergence (AND typically converges in just a few epochs given how simple it is)

Bigger picture: this is essentially the "hello world" of neural networks — it demonstrates the foundational building block (weighted sum + threshold + error-driven weight update) that more complex architectures like multi-layer perceptrons and deep networks are built on top of.
