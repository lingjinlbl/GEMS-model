import matplotlib.pyplot as plt
import numpy as np
import os

from model import Model, Optimizer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
model = Model(ROOT_DIR + "/../input-data", nSubBins=4)

operatorCosts, vectorUserCosts = model.collectAllCosts()
optimizer = Optimizer(model, modesAndMicrotypes=[('A', 'bus')], fromToSubNetworkIDs=[('A', 'Bus')], method="noisy") #,

headways = np.linspace(120, 1500, 13)
allocations = np.linspace(0.0, 0.5, 15)
collectedUserCosts = np.zeros((len(headways), len(allocations)))
collectedOperatorCosts = np.zeros((len(headways), len(allocations)))

for i, h in enumerate(headways):
    for j, a in enumerate(allocations):
        optimizer.updateAndRunModel(np.array([a, h]))
        operatorCosts, vectorUserCosts = model.collectAllCosts()
        collectedUserCosts[i, j] = vectorUserCosts.sum()
        collectedOperatorCosts[i, j] = operatorCosts.total
        print(model.getModeSpeeds())


print('done')