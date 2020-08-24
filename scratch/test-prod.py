import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



import sys
sys.path.append("/Users/zaneedell/Desktop/git/task-3-modeling")

from model import Model

spds = dict()
modesplits = dict()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
a = Model(ROOT_DIR + "/../input-data-production")
original = 41328699.75074827
dist = 1.0 * original

otherOriginal = 64110779.85721326
otherdist = 1.0 * otherOriginal

for timePeriod in ['morning_rush','other_time','evening_rush']:
    a.initializeTimePeriod(timePeriod)
    a.scenarioData['subNetworkData'].at[3, "Length"] = dist
    a.scenarioData['subNetworkData'].at[1, "Length"] = original - dist
    a.scenarioData['subNetworkData'].at[9, "Length"] = otherdist
    a.scenarioData['subNetworkData'].at[7, "Length"] = otherOriginal - otherdist
    a.findEquilibrium()
    spds[timePeriod] = a.getModeSpeeds()
    modesplits[timePeriod] = pd.DataFrame(a.getModeSplit().toDict(), index=["Aggregate"])

all = pd.concat(spds)
all.columns = pd.MultiIndex.from_tuples(all.columns.to_series().apply(lambda x: (x[0], x[2])))
all.to_csv("speeds-1600-100pctbuslane-fixed.csv")
# for g in all.columns.get_level_values(0).unique():
#     plt.plot(all[g].loc[("morning_rush","bus"),:])

popGroups = a.scenarioData["populations"]["PopulationGroupTypeID"].unique()
microtypes = a.scenarioData["populations"]["MicrotypeID"].unique()
dbins = a.scenarioData["distanceBins"]["DistanceBinID"].unique()

allModeSplits = dict()
for timePeriod in a.scenarioData["timePeriods"].TimePeriodID.values:
    for popGroup in popGroups:
        allModeSplits[popGroup + '_' + timePeriod] = pd.DataFrame(a.getModeSplit(timePeriod=timePeriod, userClass=popGroup).toDict(),index=["popGroup"])

for timePeriod in a.scenarioData["timePeriods"].TimePeriodID.values:
    for microtype in microtypes:
        allModeSplits[microtype + '_' + timePeriod] = pd.DataFrame(a.getModeSplit(timePeriod=timePeriod, microtypeID=microtype).toDict(), index=["microtype"])
        for popGroup in popGroups:
            allModeSplits[popGroup + '_' + microtype + '_' + timePeriod] = pd.DataFrame(a.getModeSplit(timePeriod=timePeriod, userClass=popGroup,microtypeID=microtype).toDict(), index=["popGroupMicrotype"])

for timePeriod in a.scenarioData["timePeriods"].TimePeriodID.values:
    for dbin in dbins:
        allModeSplits[dbin + '_' + timePeriod] = pd.DataFrame(a.getModeSplit(timePeriod=timePeriod, distanceBin=dbin).toDict(), index=["dbin"])

joined = pd.concat(allModeSplits)
joined.to_csv("groupModeSplits-1600-100pctbuslane-fixed.csv")
pd.concat(modesplits).to_csv("modeSplits-1600-100pctbuslane-fixed.csv")
print("done")