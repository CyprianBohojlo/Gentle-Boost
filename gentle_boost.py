

import numpy as np
import matplotlib.pyplot as plt
import torchvision

def fit_stump(X, Y, w, gamma):

    projections = X @ w
    a = 0
    b = 0
    c = 0
    min_error = float('inf')

    #chooses the midpoint
    unique_projections = np.unique(projections)
    thresholds = (unique_projections[:-1] + unique_projections[1:]) / 2

    #errors = []
    for b in thresholds:
        # Indicator function
        I = (projections + b > 0).astype(float)

        a = (np.sum(gamma * Y * I ) - np.sum(gamma * c * I))/ (np.sum(gamma * I) + 1e-10)

        c = (np.sum(gamma * Y) - np.sum(gamma * a * I))/ (np.sum(gamma) + 1e-10)

        error = np.sum(gamma * (Y - (a * I + c))**2)
        
        if error < min_error:
            min_error = error
            optimal_a = a
            optimal_b = b
            optimal_c = c

    return optimal_a, optimal_b, optimal_c, min_error

def gentle_boost(X, Y, k):
    n= X.shape[0]
    d = X.shape[1]
    gamma = np.ones(n) / n

    W = np.zeros((d, k))
    a_param = np.zeros(k)
    b_param = np.zeros(k)
    c_param = np.zeros(k)

    for t in range(k):
        #prints eveyr 50th iteration so it's easier to follow the process
        print(f"Current k: {t+1}/{k}") if (t + 1) % 50 == 0 or t == k - 1 else None
        w = np.random.randn(d)
        w /= np.linalg.norm(w)

        a, b, c, min_error = fit_stump(X, Y, w, gamma)

        W[:, t] = w
        a_param[t] = a
        b_param[t] = b
        c_param[t] = c

        # Update weights
        fun_x = a * (X @ w + b > 0) + c
        gamma *= np.exp(-Y * fun_x)
        gamma /= np.sum(gamma)

    return W, a_param, b_param, c_param

trainset = torchvision.datasets.USPS(root='./data', download=True, train=True)
X_train, Y_train = np.array(trainset.data) / 255., np.array(trainset.targets)

tstset = torchvision.datasets.USPS(root='./data', download=True, train=False)
X_test, Y_test = np.array(tstset.data) / 255., np.array(tstset.targets)

# Reshape the data 
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]

# Filter the data for digits 0 and 1
mask_train = ((Y_train == 0) | (Y_train == 1)).reshape(-1)
X_train, Y_train = X_train[mask_train], Y_train[mask_train]
mask_test = ((Y_test == 0) | (Y_test == 1)).reshape(-1)
X_test, Y_test = X_test[mask_test], Y_test[mask_test]

# Change labels 0 to -1 for binary classification
Y_train[Y_train == 0] = -1
Y_test[Y_test == 0] = -1
#Reshape, otherwise there dimension problem again
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
# Shuffle the training data
inds = np.random.permutation(X_train.shape[0])
X_train, Y_train = X_train[inds], Y_train[inds]

#testing data
np.unique(Y_train)
np.unique(Y_test)

#testing dimensions
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

k = 1000
W, a_param, b_param, c_param = gentle_boost(X_train, Y_train, k)

#testing if preds work
pred = X_train @ W
np.sign(pred)

train_errors = []
test_errors = []

for t in range(k):
    predictions_train = np.sum([a_param[i] * ((X_train @ W[:, i] + b_param[i]) > 0) + c_param[i] for i in range(t+1)], axis=0)
    train_error = np.mean(np.sign(predictions_train) != Y_train)
    train_errors.append(train_error)

    predictions_test = np.sum([a_param[i] * ((X_test @ W[:, i] + b_param[i]) > 0) + c_param[i] for i in range(t+1)], axis=0)
    test_error = np.mean(np.sign(predictions_test) != Y_test)
    test_errors.append(test_error)

# Plot the training and test errors
plt.figure(figsize=(10, 6))
plt.plot(range(k), train_errors, label='Training Error')
plt.plot(range(k), test_errors, label='Test Error')
plt.xlabel('Number of Iterations')
plt.ylabel('Error')
plt.xscale('log')
plt.title('Training and Test Error vs. Number of Iterations')
plt.legend()
plt.show()

