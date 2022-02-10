import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import Model, Optimizer

model = Model("input-data", nSubBins=2)
optimizer = Optimizer(model, modesAndMicrotypes=[('A', 'Bus')],  # These determine headways
                      fromToSubNetworkIDs=[('A', 'Bus')],
                      # These determine ROW allocation
                      method="min")

headways = np.linspace(60, 300, 13)
allocations = np.linspace(0., 0.62, 13)
totalUserCosts = np.zeros((len(headways), len(allocations)))
totalOperatorCosts = np.zeros((len(headways), len(allocations)))
totalOperatorRevenues = np.zeros((len(headways), len(allocations)))
netOperatorCosts = np.zeros((len(headways), len(allocations)))
totalExternalityCosts = np.zeros((len(headways), len(allocations)))
totalDedicationCosts = np.zeros((len(headways), len(allocations)))
busModeSplit = np.zeros((len(headways), len(allocations)))
carModeSplit = np.zeros((len(headways), len(allocations)))

for i, h in enumerate(headways):
    for j, a in enumerate(allocations):
        optimizer.updateAndRunModel(np.array([a, h]))
        if model.successful:
            operatorCosts, freightOperatorCosts, vectorUserCosts, externalities = model.collectAllCosts()
            allCosts = optimizer.sumAllCosts()
            dedicationCosts = model.getDedicationCostByMicrotype()
            operatorCosts = operatorCosts.toDataFrame()
            totalUserCosts[i, j] = allCosts['User'].sum()
            totalOperatorCosts[i, j] = allCosts['Operator'].sum()
            totalOperatorRevenues[i, j] = allCosts['Revenue'].sum()
            netOperatorCosts[i, j] = allCosts['Operator'].sum() - allCosts['Revenue'].sum()
            totalExternalityCosts[i, j] = allCosts['Externality'].sum()
            busModeSplit[i, j] = model.getModeSplit(microtypeID='A')[model.modeToIdx['bus']]
            carModeSplit[i, j] = model.getModeSplit(microtypeID='A')[model.modeToIdx['auto']]
            totalDedicationCosts[i, j] = allCosts['Dedication'].sum()
        else:
            print("Failed for ", np.array([a, h]))

plt.subplot(221)
plt.contourf(allocations, headways, totalUserCosts, 30, cmap='viridis')
cbar1 = plt.colorbar()
cbar1.set_label("User costs")
plt.subplot(222)
plt.contourf(allocations, headways, netOperatorCosts, 30, cmap='plasma')
cbar2 = plt.colorbar()
cbar2.set_label("Operator costs")
plt.subplot(223)
plt.contourf(allocations, headways, totalExternalityCosts, 30, cmap='inferno')
cbar3 = plt.colorbar()
cbar3.set_label("Externality costs")
plt.subplot(224)
plt.contourf(allocations, headways, totalUserCosts + netOperatorCosts + totalExternalityCosts + totalDedicationCosts,
             30,
             cmap='cividis')
cbar4 = plt.colorbar()
cbar4.set_label("Total costs")

# // Run the model on LA with default parameters
model = Model("input-data-losangeles-national-params", nSubBins=1)
optimizer = Optimizer(model, modesAndMicrotypes=[('1', 'Bus'), ('2', 'Bus')],  # These determine headways
                      fromToSubNetworkIDs=[('1', 'Bus'), ('2', 'Bus'), ('1', 'Bike'), ('2', 'Bike')],
                      # These determine ROW allocation
                      method="min")

x0 = optimizer.x0()

# x0 holds the decision variables. In this example, they are
# [Portion bus ROW in Microtype 1, bike Row in 1, Bus ROW in 2, Bike ROW in 2, Bus headway
# in 1 (in seconds / 100), Bus headway in 2)].
# These are defined in the Optimizer definition above

optimizer.evaluate(x0)
modeSplit, speed, utility = model.toPandas()
allCosts = optimizer.sumAllCosts()

# // Update the model to allocate 10% of bus route to bus lanes in microtype 1
optimizer.evaluate([0.1, 0., 0., 0., 0.3, 0.3])
modeSplit2, speed2, utility2 = model.toPandas()
allCosts2 = optimizer.sumAllCosts()

# // Run the full optimizer (this can take a long time!)
optimizer.minimize()
modeSplit_opt, speed_opt, utility_opt = model.toPandas()
allCosts_opt = optimizer.sumAllCosts()
print('done')
