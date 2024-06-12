import numpy as np

trainExamples = [
    (1,1),
    (2,3),
    (4,3)
]

def phi(x):
    return np.array([1,x])

def initialWeightVector():
    return np.zeros(2)

def trainLoss(w):
    return 1.0 / len(trainExamples) * sum((w.dot(phi(x))-y)**2 for x, y in trainExamples)

def gradientTrainLoss(w):
    return 1.0 / len(trainExamples) * sum(2 * (w.dot(phi(x))-y) * phi(x) for x, y in trainExamples)

def gradientDescent(F, gradientF, initialWeightVector):
    w = initialWeightVector()
    eta = 0.1

    for i in range(500):
        value = F(w)
        gradient = gradientF(w)
        w -= eta * gradient
        print(f"epoch {i}: w = {w}, F(w) = {value}, gradientF = {gradient}")


gradientDescent(trainLoss, gradientTrainLoss, initialWeightVector)


