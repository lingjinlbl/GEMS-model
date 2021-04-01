import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import Model

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
a = Model(ROOT_DIR + "/../input-data-geotype-A")
# userCosts, operatorCosts = a.collectAllCosts()
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
sn = pd.read_csv(ROOT_DIR + "/../input-data-geotype-A/SubNetworks.csv", index_col="SubnetworkID",
                 dtype={"MicrotypeID": str})
mask = (sn.ModesAllowed == 'Auto-Bus-Bike')
jamDensity = np.linspace(0.12, 0.2, num=20)
fig1 = plt.figure()
ax1, ax2 = fig1.subplots(1, 2)
i = 0
for den in jamDensity:
    a.scenarioData['subNetworkData'].loc[mask, 'densityMax'] = den
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
    print(ms)
    allCosts.append(userCosts.total + operatorCosts.total)
    if i == 0:
        x, y = a.plotAllDynamicStats("density")
        ax1.plot(x, y)
    i += 1
x, y = a.plotAllDynamicStats("density")
ax2.plot(x, y)
allSpeeds = pd.concat(allSpeeds, keys=jamDensity, names=['jamDensity']).swaplevel(0, 1)

fig2 = plt.figure()
ax21, ax22 = fig2.subplots(1, 2)
allSpeeds.loc['bus'].plot(ax=ax21, legend=False)
allSpeeds.loc['auto'].plot(ax=ax22, legend=False)
fig2.legend(a.getMicrotypeCollection(0).microtypeNames(), title="Microtype")
ax21.set_ylabel("Bus speed (m/s)")
ax22.set_ylabel("Auto speed (m/s)")

plt.xlabel("Bus Lane Distance In Microtype B")
plt.ylabel("Bus Speeds")

everything = pd.concat(allCostObjects, keys=jamDensity, names=['jamDensity'])
everything.groupby(['jamDensity', 'mode']).agg('sum')['totalCost'].unstack().plot()

busTrips = everything.groupby(['jamDensity', 'mode', 'homeMicrotype']).agg('sum').unstack(level=1)[
    'demandForTripsPerHour', 'bus'].unstack()
allTrips = everything.groupby(['jamDensity', 'mode', 'homeMicrotype']).agg('sum').unstack(level=1)[
    'demandForTripsPerHour']

busModeSplit = busTrips / allTrips.sum(axis=1).unstack()
tt = df.groupby(['busLaneDistance','populationGroupType','tripPurpose']).agg(sum).loc[0]['travelTime'].unstack()
y = -everything.groupby('jamDensity').agg(sum)['totalCost'].values
trips = df.groupby(['busLaneDistance','homeMicrotype','mode']).agg(sum).loc[0]['demandForTripsPerHour'].unstack()
print("DONE")
