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
original = 3916704.6807775586
dist = 201670.0

otherOriginal = 6075753.0

for timePeriod in ['morning_rush','other_time','evening_rush']:
    a.initializeTimePeriod(timePeriod)
    a.scenarioData['subNetworkData'].at[3, "Length"] = dist
    a.scenarioData['subNetworkData'].at[1, "Length"] = original - dist
    a.scenarioData['subNetworkData'].at[9, "Length"] = dist
    a.scenarioData['subNetworkData'].at[7, "Length"] = otherOriginal - dist
    a.findEquilibrium()
    spds[timePeriod] = a.getModeSpeeds()
    modesplits[timePeriod] = pd.DataFrame(a.getModeSplit().toDict(), index=["Aggregate"])

all = pd.concat(spds)
all.columns = pd.MultiIndex.from_tuples(all.columns.to_series().apply(lambda x: (x[0], x[2])))
all.to_csv("speeds-1200-5pctbuslaneA1A2-hw2.csv")
# for g in all.columns.get_level_values(0).unique():
#     plt.plot(all[g].loc[("morning_rush","bus"),:])

popGroups = a.scenarioData["populations"]["PopulationGroupTypeID"].unique()
microtypes = a.scenarioData["populations"]["MicrotypeID"].unique()
dbins = a.scenarioData["distanceBins"]["DistanceBinID"].unique()

allModeSplits = dict()
for popGroup in popGroups:
    allModeSplits[popGroup] = pd.DataFrame(a.getModeSplit(userClass=popGroup).toDict(),index=["popGroup"])

for microtype in microtypes:
    allModeSplits[microtype] = pd.DataFrame(a.getModeSplit(microtypeID=microtype).toDict(), index=["microtype"])
    for popGroup in popGroups:
        allModeSplits[popGroup + '_' + microtype] = pd.DataFrame(a.getModeSplit(userClass=popGroup,microtypeID=microtype).toDict(), index=["popGroupMicrotype"])

for dbin in dbins:
    allModeSplits[dbin] = pd.DataFrame(a.getModeSplit(distanceBin=dbin).toDict(), index=["dbin"])

joined = pd.concat(allModeSplits)
joined.to_csv("groupModeSplits-1200-5pctbuslaneA1A2-hw2.csv")
pd.concat(modesplits).to_csv("modeSplits-1200-5pctbuslaneA1A2-hw2.csv")
print("done")