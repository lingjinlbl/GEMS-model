import matplotlib.pyplot as plt
import numpy as np
import os

from model import Model, Optimizer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
model = Model(ROOT_DIR + "/../input-data", nSubBins=2)

operatorCosts, vectorUserCosts, externalities = model.collectAllCosts()
optimizer = Optimizer(model, modesAndMicrotypes=[('A', 'bus')], fromToSubNetworkIDs=[('A', 'bus')], method="noisy")  # ,

headways = np.linspace(60, 300, 10)
allocations = np.linspace(0.1, 0.2, 10)
collectedUserCosts = np.zeros((len(headways), len(allocations)))
collectedOperatorCosts = np.zeros((len(headways), len(allocations)))
collectedExternalityCosts = np.zeros((len(headways), len(allocations)))
busModeSplit = np.zeros((len(headways), len(allocations)))
carModeSplit = np.zeros((len(headways), len(allocations)))

for i, h in enumerate(headways):
    for j, a in enumerate(allocations):
        optimizer.updateAndRunModel(np.array([a, h]))
        if model.successful:
            operatorCosts, vectorUserCosts, externalities = model.collectAllCosts()  # TODO: add back in lane dedication
            collectedUserCosts[i, j] = sum([uc.sum() for uc in vectorUserCosts.values()])
            collectedOperatorCosts[i, j] = operatorCosts.total
            collectedExternalityCosts[i, j] = sum([ex.sum() for ex in externalities.values()])
            busModeSplit[i, j] = model.getModeSplit()[model.modeToIdx['bus']]
            carModeSplit[i, j] = model.getModeSplit()[model.modeToIdx['auto']]
        else:
            print("Failed for ", np.array([a, h]))
            collectedUserCosts[i, j] = np.nan
            collectedOperatorCosts[i, j] = np.nan
            collectedExternalityCosts[i, j] = np.nan
            busModeSplit[i, j] = np.nan
            carModeSplit[i, j] = np.nan
        print(model.getModeSpeeds())

print('done')
plt.subplot(221)
plt.contourf(allocations, headways, collectedUserCosts, 30, cmap='viridis')
cbar1 = plt.colorbar()
cbar1.set_label("User costs")
plt.subplot(222)
plt.contourf(allocations, headways, collectedOperatorCosts, 30, cmap='plasma')
cbar2 = plt.colorbar()
cbar2.set_label("Operator costs")
plt.subplot(223)
plt.contourf(allocations, headways, collectedExternalityCosts, 30, cmap='inferno')
cbar3 = plt.colorbar()
cbar3.set_label("Externality costs")
plt.subplot(224)
plt.contourf(allocations, headways, collectedExternalityCosts + collectedOperatorCosts + collectedUserCosts, 30,
             cmap='cividis')
cbar4 = plt.colorbar()
cbar4.set_label("Total costs")

print('done')
