def f_true(x):
  y = 6.0 * (np.sin(x + 2) + np.sin(2*x + 4))
  return y

import numpy as np                       # For all our math needs
n = 750                                  # Number of data points
X = np.random.uniform(-7.5, 7.5, n)      # Training examples, in one dimension
e = np.random.normal(0.0, 5.0, n)        # Random Gaussian noise
y = f_true(X) + e                        # True labels with noise

import matplotlib.pyplot as plt          # For all our plotting needs
plt.figure()

plt.scatter(X, y, 12, marker='o')           

x_true = np.arange(-7.5, 7.5, 0.05)
y_true = f_true(x_true)
plt.plot(x_true, y_true, marker='None', color='r')

from sklearn.model_selection import train_test_split
tst_frac = 0.3  # Fraction of examples to sample for the test set
val_frac = 0.1  # Fraction of examples to sample for the validation set

# First, we use train_test_split to partition (X, y) into training and test sets
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)

# Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)

# Plot the three subsets
plt.figure()
plt.scatter(X_trn, y_trn, 12, marker='o', color='orange')
plt.scatter(X_val, y_val, 12, marker='o', color='green')
plt.scatter(X_tst, y_tst, 12, marker='o', color='blue')

def polynomial_transform(X, d):
    Phi = []
    for value in X:
        z = []
        for d in range(0, d+1):
            z.append(np.power(value, d))
        Phi.append(z)
    Phi = np.asarray(Phi)
    return Phi
    
def train_model(Phi, y):
    w = np.linalg.inv(np.transpose(Phi) @ Phi) @ np.transpose(Phi) @ y
    return w

def evaluate_model(Phi, y, w):
    #ypred = Phi @ w
    err = ((Phi @ w) - y)**2
    #err = (y - (Phi @ w))**2
    sum = 0
    n = np.size(y)
    for value in err:
        sum = sum + value
        #n = n + 1
    meansq = sum / n
    return meansq

w = {}               # Dictionary to store all the trained models
validationErr = {}   # Validation error of the models
testErr = {}         # Test error of all the models

for d in range(3, 25, 3):  # Iterate over polynomial degree
    Phi_trn = polynomial_transform(X_trn, d)                 # Transform training data into d dimensions
    w[d] = train_model(Phi_trn, y_trn)                       # Learn model on training data
    
    Phi_val = polynomial_transform(X_val, d)                 # Transform validation data into d dimensions
    validationErr[d] = evaluate_model(Phi_val, y_val, w[d])  # Evaluate model on validation data
    
    Phi_tst = polynomial_transform(X_tst, d)           # Transform test data into d dimensions
    testErr[d] = evaluate_model(Phi_tst, y_tst, w[d])  # Evaluate model on test data

plt.figure()
plt.plot(validationErr.keys(), validationErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testErr.keys(), testErr.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Polynomial degree', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([2, 25, 15, 60])

print("The Validation and Test Error drop consistently until about a polynomial degree of 15 to 18, so any degree between these two numbers would yield the best accuracy in my opinion")

plt.show()

plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for d in range(9, 25, 3):
    X_d = polynomial_transform(x_true, d)
    y_d = X_d @ w[d]

    plt.plot(x_true, y_d, marker='None', linewidth=2)

plt.legend(['true'] + list(range(9, 25, 3)))
plt.axis([-8, 8, -15, 15])

plt.show()

import math
def radial_basis_transform(X, B, gamma=0.1):
    Phi = []
    for value in X:
        z = []
        for a in B:#range(0, np.size(X)):
            z.append(math.e ** (-1 * gamma * (value - a) ** 2))#or neg gamma not sure
        Phi.append(z)
    Phi = np.array(Phi)
    #Phi = Phi.reshape(np.size(X), np.size(B))
    return Phi

def train_ridge_model(Phi, y, lam):
    n = np.shape(Phi)[0]
    w = np.linalg.inv(np.transpose(Phi) @ Phi + lam * np.eye(n)) @ np.transpose(Phi) @ y
    return w

q = {}               # Dictionary to store all the trained models
validationError = {}   # Validation error of the models
testError = {}         # Test error of all the models
d = 10 ** -3

#for d in range(np.power(10, -3), np.power(10, 3), step):  # Iterate over polynomial degree
while d <= 10 ** 3:
    Phi_train = radial_basis_transform(X_trn, X_trn)                # Transform training data into d dimensions
    q[d] = train_ridge_model(Phi_train, y_trn, d)                       # Learn model on training data

    Phi_value = radial_basis_transform(X_val, X_trn)                # Transform validation data into d dimensions
    validationError[d] = evaluate_model(Phi_value, y_val, q[d])  # Evaluate model on validation data

    Phi_test = radial_basis_transform(X_tst, X_trn)        # Transform test data into d dimensions
    testError[d] = evaluate_model(Phi_test, y_tst, q[d])  # Evaluate model on test data
    d = d * 10 

# Plot all the models
plt.figure()
plt.plot(validationError.keys(), validationError.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testError.keys(), testError.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Gamma', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationError.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([0, 0.01, 20, 65])
plt.show()

plt.figure()
plt.plot(validationError.keys(), validationError.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testError.keys(), testError.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Gamma', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationError.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([0, 1, 20, 65])
plt.show()

plt.figure()
plt.plot(validationError.keys(), validationError.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testError.keys(), testError.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Regularization Parameter', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationError.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([0, 1000, 20, 65])
plt.show()

print("It seems that the validation and test error relative to the regularization paramater have a pattern similar to a log function, thus the most optimal value for lambda would be 10^-3 or 10^-2 as both the test and validation error are lowest then.")


plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

d = 10 ** -3

while d <= 10 ** 3:
    X_b = radial_basis_transform(x_true, X_trn)
    y_b = X_b @ q[d]
    plt.plot(x_true, y_b, marker='None', linewidth=2)
    d = d * 10

a = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
plt.legend(['true'] + a)
plt.show()
