import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import Model

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
model = Model(ROOT_DIR + "/../input-data-simpler")
model.scenarioData['populationGroups'].loc[
    model.scenarioData['populationGroups']['Mode'] == "bus", "BetaTravelTime"] = -0.02
model.scenarioData['populationGroups'].loc[
    model.scenarioData['populationGroups']['Mode'] == "auto", "BetaTravelTime"] = -0.02
model.readFiles()
# Redo this
fig, axs = plt.subplots(4, 4)
demands = [160, 200, 240, 280]
# demands = [390, 400, 405, 410]
totalTimesPlot = []
totalTimesSpeed = []
for ind, d in enumerate(demands):
    model.updateTimePeriodDemand(1, d)
    vectorUserCosts, utils = model.collectAllCharacteristics()
    x, y = model.plotAllDynamicStats("delay")
    axs[0, ind].clear()
    axs[0, ind].set_title("Demand: " + str(d))
    axs[0, ind].plot(x, y)

    axs[1, ind].clear()
    axs[1, ind].plot(x[:-1], np.interp(y[:, 0], y[:, 1], x)[:-1]/60. - x[:-1]/60., 'C3')
    axs[1, ind].set_ylim((0, 50))
    axs[1, ind].step(np.array([0, 1, 2, 3])*3609, np.vstack(
        [6436/model.getModeSpeeds(0)['auto']/60, 6436/model.getModeSpeeds(0)['auto']/60, 6436/model.getModeSpeeds(1)['auto']/60,
         6436/model.getModeSpeeds(2)['auto']/60]), color='C4')
    # axs[1, ind].plot(x[1:], np.diff(y, axis=0))
    # axs[1, ind].step(np.array([0, 1, 2, 3]) * 3600, np.vstack(
    #     [6436 / model.getModeSpeeds(0)['bus'], 6436 / model.getModeSpeeds(0)['bus'],
    #      6436 / model.getModeSpeeds(1)['bus'], 6436 / model.getModeSpeeds(2)['bus']]))

    axs[2, ind].clear()
    # axs[2, ind].plot(x, y[:, 0] - y[:, 1])
    axs[2, ind].step([0, 1, 2, 3], np.vstack([model.getModeSplit(0), model.getModeSplit(0), model.getModeSplit(1),
                                              model.getModeSplit(2)]))
    axs[2, ind].set_ylim([0, 1])

    axs[2, ind].lines[model.modeToIdx['auto']].set_color('C5')
    axs[2, ind].lines[model.modeToIdx['bus']].set_color('C6')
    axs[2, ind].lines[model.modeToIdx['walk']].set_color('C7')

    axs[3, ind].clear()
    # axs[3, ind].set_ylim((200,1000))
    axs[3, ind].step(np.array([0, 1, 2, 3])*3600, np.vstack(
        [model.getModeSpeeds(0)['auto'], model.getModeSpeeds(0)['auto'], model.getModeSpeeds(1)['auto'], model.getModeSpeeds(2)['auto']]))
    # axs[1, ind].plot(x[1:], np.diff(y, axis=0))
    axs[3, ind].step(np.array([0, 1, 2, 3]) * 3600, np.vstack(
        [model.getModeSpeeds(0)['bus'], model.getModeSpeeds(0)['bus'],
         model.getModeSpeeds(1)['bus'], model.getModeSpeeds(2)['bus']]))
    axs[3, ind].step(np.array([0, 1, 2, 3]) * 3600, np.vstack(
        [model.getModeSpeeds(0)['walk'], model.getModeSpeeds(0)['walk'],
         model.getModeSpeeds(1)['walk'], model.getModeSpeeds(2)['walk']]))
    axs[3, ind].lines[model.modeToIdx['auto']].set_color('C7')
    axs[3, ind].lines[model.modeToIdx['bus']].set_color('C5')
    axs[3, ind].lines[model.modeToIdx['walk']].set_color('C6')
    totalTimesPlot.append(np.sum(y[:, 0] - y[:, 1]))
    totalTimesSpeed.append(vectorUserCosts[2, 1] * 60)
axs[0, 0].set_ylabel("Cumulative vehicles")
axs[0, 0].legend(['Arrivals', 'Departures'])
axs[1, 0].set_ylabel("Travel Time (min)")
axs[1, 0].legend(["Inst. Travel Time", "Avg. Travel Time"])
axs[2, 0].set_ylabel("Mode Split")
axs[2, 0].legend(list(model.modeToIdx.keys()))
axs[3, 0].set_ylabel("Speed")
axs.flat[0].set_ylabel("Cumulative vehicles")

print('done')
#
# fig2, axs2 = plt.subplots(4, 4)
# busVOTs = [-0.1, -0.05, -0.03, -0.02]
# busModeSplit = []
# carModeSplit = []
# for ind, vot in enumerate(busVOTs):
#     model.scenarioData['populationGroups'].loc[
#         model.scenarioData['populationGroups']['Mode'] == "bus", "BetaTravelTime"] = vot
#     model.readFiles()
#     model.updateTimePeriodDemand(1, 320)
#     vectorUserCosts = model.collectAllCharacteristics()
#     x, y = model.plotAllDynamicStats("delay")
#
#     axs2[0, ind].clear()
#     axs2[0, ind].plot(x[1:], np.diff(y, axis=0))
#     axs2[0, ind].set_title("Bus VOT: " + str(vot))
#
#     axs2[1, ind].clear()
#     axs2[1, ind].step([0, 1, 2, 3], np.vstack(
#         [model.getModeSplit(0), model.getModeSplit(0), model.getModeSplit(1), model.getModeSplit(2)]))
#     axs2[1, ind].set_ylim([0, 1])
#
#     axs2[2, ind].clear()
#     axs2[2, ind].step([0, 1, 2, 3], np.vstack(
#         [model.getModeSpeeds(0).to_numpy(), model.getModeSpeeds(0).to_numpy(), model.getModeSpeeds(1).to_numpy(),
#          model.getModeSpeeds(2).to_numpy()]))
#     axs2[2, ind].set_ylim([0, 17])
#
#     axs2[3, ind].clear()
#     axs2[3, ind].plot(['bus','auto'],model.choice.numpy[0,[0,2],:][:,[1,3,4]])
#     axs2[3, ind].set_ylim([0, 0.6])
#
#
#     busModeSplit.append(model.getModeSplit()[model.scenarioData.modeToIdx["bus"]])
#     carModeSplit.append(model.getModeSplit()[model.scenarioData.modeToIdx["auto"]])
# axs2[0, 0].set_ylabel("Rate of Change")
# axs2[0, 0].legend(['Arrivals', 'Departures'])
# axs2[1, 0].set_ylabel("Mode Split")
# axs2[1, 0].legend(['Bus', 'Walk', 'Auto'])
# axs2[2, 0].set_ylabel("Speed")
# axs2[3, 0].set_ylabel("Time (hrs)")
# axs2[3, 0].legend(['In Vehicle', 'Headway', 'Access'])

model.scenarioData['populationGroups'].loc[
        model.scenarioData['populationGroups']['Mode'] == "bus", "BetaTravelTime"] = -0.03
model.scenarioData['subNetworkData'].loc[2,:] = model.scenarioData['subNetworkData'].loc[1,:].copy()
model.scenarioData['subNetworkData'].loc[2, "Length"] = 0
newmap = model.scenarioData['modeToSubNetworkData'].iloc[1:,].copy()
newmap['SubnetworkID'] += 1
model.scenarioData['modeToSubNetworkData'] = model.scenarioData['modeToSubNetworkData'].append(newmap)
model.readFiles()
initialDistance = model.scenarioData['subNetworkData'].loc[1, "Length"]
busLaneDistance = np.linspace(0, 750, 10)

fig3, axs3 = plt.subplots(2, 4)
demands = [200, 250, 275, 300]
for ind, d in enumerate(demands):
    model.updateTimePeriodDemand(1, d)
    model.microtypes.updateNetworkData()
    busSpeed = []
    carSpeed = []
    busModeShare = []
    carModeShare = []
    for dist in busLaneDistance:
        model.scenarioData['subNetworkData'].at[2, "Length"] = dist
        model.scenarioData['subNetworkData'].at[1, "Length"] = initialDistance - dist
        model.microtypes.updateNetworkData()
        vectorUserCosts = model.collectAllCharacteristics()
        ms = model.getModeSplit()
        spd = model.getModeSpeeds()

        speeds = pd.DataFrame(model.microtypes.getModeSpeeds())
        busModeShare.append(model.demand.getTotalModeSplit()['bus'])
        carModeShare.append(model.demand.getTotalModeSplit()['auto'])
        busSpeed.append(spd.loc['A', 'bus'])
        carSpeed.append(spd.loc['A', 'auto'])
    axs3[0, ind].clear()
    axs3[0, ind].scatter(busLaneDistance, carModeShare)
    axs3[0, ind].scatter(busLaneDistance, busModeShare)
    axs3[0, ind].set_ylim((0,1))
    axs3[0, ind].set_title('Demand: ' + str(d))

    axs3[1, ind].clear()
    axs3[1, ind].scatter(busLaneDistance, carSpeed)
    axs3[1, ind].scatter(busLaneDistance, busSpeed)
    axs3[1, ind].set_ylim((0, 16))
print('done')
