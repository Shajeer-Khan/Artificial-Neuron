import numpy as np

# Set the learning rate (alpha)
alpha = 0.1

# Set the initial weights and bias
w1 = np.random.random()
w2 = np.random.random()
b = np.random.random()

# Define the activation function
def activation(x):
  if x > 0:
    return 1
  else:
    return 0

# Define the AND function using the activation function
def AND(x1, x2):
  y = activation(w1 * x1 + w2 * x2 + b)
  return y

# Train the AND function using the delta rule
for i in range(10000):
  # Generate a random input sample
  x1 = np.random.randint(2)
  x2 = np.random.randint(2)
  y = AND(x1, x2)
  
  # Calculate the error
  y_true = x1 and x2
  error = y_true - y
  
  # Adjust the weights and bias
  w1 += alpha * error * x1
  w2 += alpha * error * x2
  b += alpha * error

# Test the AND function
print(AND(0, 0))  # Expected output: 0
print(AND(0, 1))  # Expected output: 0
print(AND(1, 0))  # Expected output: 0
print(AND(1, 1))  # Expected output: 1