import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from model import Model

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
a = Model(ROOT_DIR + "/../input-data")
userCosts, operatorCosts = a.collectAllCosts()
allSpeeds = []
busModeShare = []
carModeShare = []
totalUserCosts = []
totalOperatorCosts = []
costByMode = []
demandForTripsByMode = []
demandForPMTByMode = []
ldCosts = []
allCosts = []
allCostObjects = []
orig_dist = a.scenarioData['subNetworkData'].at[1, "Length"]
busLaneDistance = np.arange(0, 4000, 250)
for dist in busLaneDistance:
    a.scenarioData['subNetworkData'].at[9, "Length"] = dist
    a.scenarioData['subNetworkData'].at[1, "Length"] = orig_dist - dist
    userCosts, operatorCosts = a.collectAllCosts()
    allCostObjects.append(userCosts.toDataFrame())
    ms = a.getModeSplit()
    speeds = pd.DataFrame(a.microtypes.getModeSpeeds())

    allSpeeds.append(speeds)
    costByMode.append(userCosts.groupBy(['mode'])['totalCost'])
    demandForTripsByMode.append(userCosts.groupBy(['mode'])['demandForTripsPerHour'])
    demandForPMTByMode.append(userCosts.groupBy(['mode'])['demandForPMTPerHour'])

    busModeShare.append(ms["bus"])
    carModeShare.append(ms["auto"])
    print("Bus lane dist: " + str(dist))
    print("Car speed: " + str([n.base_speed for n in a.microtypes["B"].networks.modes["bus"].networks]))
    print("Bus speed: " + str([n for n in a.microtypes["B"].networks.modes["bus"]._speed.values()]))
    print(ms)
    ldCosts.append(0.014 * dist)
    allCosts.append(userCosts.total + operatorCosts.total + 0.014 * dist)
allSpeeds = pd.concat(allSpeeds, keys = busLaneDistance, names = ['busLaneDistance']).swaplevel(0,1)

allSpeeds.loc['bus'].plot()

plt.xlabel("Bus Lane Distance In Microtype B")
plt.ylabel("Bus Speeds")

everything = pd.concat(allCostObjects, keys = busLaneDistance, names = ['busLaneDistance'])
everything.groupby(['busLaneDistance','mode']).agg('sum')['totalCost'].unstack().plot()