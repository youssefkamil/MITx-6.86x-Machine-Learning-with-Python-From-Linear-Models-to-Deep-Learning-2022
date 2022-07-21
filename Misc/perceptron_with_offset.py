#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:59:45 2019

@author: dileepn

Perceptron algorithm with offset
"""
import numpy as np
import matplotlib.pyplot as plt

# Data points
x = np.array([[0,0],[2,0],[3,0],[0,2],[2,2],[5,1],[5,2],[2,4],[4,4],[5,5]])
#x = np.array([[-4,2],[-2,1],[-1,-1],[2,2],[1,-2]])
#x = np.array([[1,0],[-1,10],[-1,-1]])

# Labels
y = np.array([[-1],[-1],[-1],[-1],[-1],[1],[1],[1],[1],[1]])
#y = np.array([[1],[1],[-1],[-1],[-1]])
#y = np.array([[-1],[1],[1]])

# Plot data
colors = ['b' if y == 1 else 'r' for y in y]
plt.figure()
plt.scatter(x[:,0], x[:,1], s=40, c=colors)

# Number of examples
n = x.shape[0]

# Number of features
m = x.shape[1]

# No. of iterations
T = 10

# Initialize parameter vector and offset
theta = np.array([[1],[1]])
theta0 = -5

# Tolerance for floating point errors
eps = 1e-8

# Start the perceptron update loop
mistakes = 0    # Keep track of mistakes
for t in range(T):
    counter = 0     # To check if all examples are classified correctly in loop
    for i in range(n):
        agreement = float(y[i]*(theta.T.dot(x[i,:]) + theta0))
        if abs(agreement) < eps or agreement < 0.0:
            theta = theta + y[i]*x[i,:].reshape((m,1))
            theta0 = theta0 + float(y[i])
            print("current parameter vector:", theta)
            print("current offset: {:.1f}".format(theta0))
            mistakes += 1
        else:
            counter += 1
    
    # If all examples classified correctly, stop
    if counter == n:
        print("No. of iteration loops through the dataset:", t+1)
        break
    
# Print total number of mistakes
print("Total number of misclassifications:", mistakes)

# Plot the decision boundary
x_line = np.linspace(-1,6,100)
y_line = (-theta0 - theta[0]*x_line)/theta[1]
y_line2 = (18 - 4*x_line)/4
plt.plot(x_line, y_line, 'k-', linewidth = 2, label = 'Max. Margin Separator')
plt.plot(x_line, y_line2, 'g--', linewidth = 1, label = 'Perceptron Solution')
plt.legend()