import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



import sys
sys.path.append("/Users/zaneedell/Desktop/git/task-3-modeling")

from model import Model

for factor in range(0,110,10):
    spds = dict()
    modesplits = dict()
    userCosts = dict()
    operatorCosts = dict()

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    a = Model(ROOT_DIR + "/../input-data-production")
    original = a.scenarioData['subNetworkData'].at[43, "Length"]
    dist = factor * original / 100.
    string = str(factor) + "pctbuslane.csv"

    for timePeriod in ['morning_rush','other_time','evening_rush']:
        a.initializeTimePeriod(timePeriod)
        a.scenarioData['subNetworkData'].at[45, "Length"] = dist
        a.scenarioData['subNetworkData'].at[43, "Length"] = original - dist
        a.findEquilibrium()
        spds[timePeriod] = a.getModeSpeeds()
        modesplits[timePeriod] = pd.DataFrame(a.getModeSplit().toDict(), index=["Aggregate"])
        userCosts[timePeriod] = a.getModeUserCosts()
        operatorCosts[timePeriod] = a.getOperatorCosts().toDataFrame()

    all = pd.concat(spds)
    all.columns = pd.MultiIndex.from_tuples(all.columns.to_series().apply(lambda x: (x[0], x[2])))
    all.to_csv("out/B_speeds-" + string)

    pd.concat(userCosts).to_csv("out/B_userCosts-"+string)
    pd.concat(operatorCosts).to_csv("out/B_operatorCosts-"+string)
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
    joined.to_csv("out/B_groupModeSplits-"+string)
    pd.concat(modesplits).to_csv("out/B_modeSplits-"+string)