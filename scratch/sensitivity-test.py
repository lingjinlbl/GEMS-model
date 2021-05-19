import os

import matplotlib.pyplot as plt
import numpy as np

from model import Model

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
model = Model(ROOT_DIR + "/../input-data-simpler")
model.scenarioData['populationGroups'].loc[
    model.scenarioData['populationGroups']['Mode'] == "bus", "BetaTravelTime"] = -1e9
model.readFiles()
fig, axs = plt.subplots(3, 5)
demands = [270, 280, 290, 300, 305]
totalTimesPlot = []
totalTimesSpeed = []
for ind, d in enumerate(demands):
    model.updateTimePeriodDemand(1, d)
    vectorUserCosts = model.collectAllCharacteristics()
    x, y = model.plotAllDynamicStats("delay")
    axs[0, ind].clear()
    axs[0, ind].set_title("Demand: " + str(d))
    axs[0, ind].plot(x, y)

    axs[1, ind].clear()
    axs[1, ind].plot(x[1:], np.diff(y, axis=0))

    axs[2, ind].clear()
    axs[2, ind].plot(x, y[:, 0] - y[:, 1])
    totalTimesPlot.append(np.sum(y[:, 0] - y[:, 1]))
    totalTimesSpeed.append(vectorUserCosts[2, 1] * 60)
axs[0, 0].set_ylabel("Cumulative vehicles")
axs[0, 0].legend(['Arrivals', 'Departures'])
axs[1, 0].set_ylabel("Rate of Change")
axs[1, 0].legend(["Inflow", "Outflow"])
axs[2, 0].set_ylabel("Accumulation")
axs.flat[0].set_ylabel("Cumulative vehicles")

fig2, axs2 = plt.subplots(3, 4)
busVOTs = [-0.1, -0.05, -0.03, -0.02]
busModeSplit = []
carModeSplit = []
for ind, vot in enumerate(busVOTs):
    model.scenarioData['populationGroups'].loc[
        model.scenarioData['populationGroups']['Mode'] == "bus", "BetaTravelTime"] = vot
    model.readFiles()
    model.updateTimePeriodDemand(1, 320)
    vectorUserCosts = model.collectAllCharacteristics()
    x, y = model.plotAllDynamicStats("delay")

    axs2[0, ind].clear()
    axs2[0, ind].plot(x[1:], np.diff(y, axis=0))

    axs2[1, ind].clear()
    axs2[1, ind].step([0, 1, 2, 3], np.vstack(
        [model.getModeSplit(0), model.getModeSplit(0), model.getModeSplit(1), model.getModeSplit(2)]))
    axs2[1, ind].set_ylim([0, 1])

    axs2[2, ind].clear()
    axs2[2, ind].step([0, 1, 2, 3], np.vstack(
        [model.getModeSpeeds(0).to_numpy(), model.getModeSpeeds(0).to_numpy(), model.getModeSpeeds(1).to_numpy(),
         model.getModeSpeeds(2).to_numpy()]))
    axs2[2, ind].set_ylim([0, 17])
    busModeSplit.append(model.getModeSplit()[model.scenarioData.modeToIdx["bus"]])
    carModeSplit.append(model.getModeSplit()[model.scenarioData.modeToIdx["auto"]])
axs2[0, 0].set_ylabel("Rate of Change")
axs2[0, 0].legend(['Bus', 'Walk', 'Auto'])
axs2[1, 0].set_ylabel("Mode Split")
axs2[2, 0].set_ylabel("Speed")
print('done')
