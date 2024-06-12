import numpy as np

trainExamples = [
    ((0,2), 1),
    ((-2,0), 1),
    ((1,-1), -1)
    ]

def phi(x):
    return np.array(x)

def initialWeightVector():
    return np.zeros(2)

def trainLoss(w):
    return 1.0 / len(trainExamples) * sum(max(1 - w.dot(phi(x)) * y, 0) for x, y in trainExamples)

def gradientTrainLoss(w):
    return 1.0 / len(trainExamples) * sum(-phi(x) * y if 1 - w.dot(phi(x)) * y > 0 else 0 for x, y in trainExamples)

def gradientDescent(F, gradientF, initialWeightVector):
    w = initialWeightVector()
    eta = 0.1

    for i in range(500):
        value = F(w)
        gradient = gradientF(w)
        w -= eta * gradient
        print(f"epoch {i}: w = {w}, F(w) = {value}, gradientF = {gradient}")


gradientDescent(trainLoss, gradientTrainLoss, initialWeightVector)


