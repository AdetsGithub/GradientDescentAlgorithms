import numpy as np
import math

trueW = np.array([1,2,3,4,5])

def generate():
    x = np.random.randn(len(trueW))
    y = trueW.dot(x) + np.random.randn()
    # print(x, y)
    return (x,y)

trainExamples = [generate() for i in range (1000000)]

def phi(x):
    return np.array(x)

def initialWeightVector():
    return np.zeros(len(trueW))

def loss(w, i):
    x, y = trainExamples[i]
    return (w.dot(phi(x)) - y)**2

def gradientLoss(w, i):
    x, y = trainExamples[i]
    return 2 * (w.dot(phi(x)) - y) * phi(x)

def stochasticGradientDescent(f, gradientf, n, initialWeightVector):
    w = initialWeightVector()
    numUpdates = 0
    for i in range(500):
        for j in range(n):
            value = f(w, j)
            gradient = gradientf(w, j)
            numUpdates += 1
            eta = 1.0 / math.sqrt(numUpdates)
            w -= eta * gradient             
        print(f"epoch {i}: w = {w}, F(w) = {value}, gradientF = {gradient}")


stochasticGradientDescent(loss, gradientLoss, len(trainExamples), initialWeightVector)