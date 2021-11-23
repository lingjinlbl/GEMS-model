import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import Model


def test_find_equilibrium():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    a = Model(ROOT_DIR + "/../input-data")
    a.initializeTimePeriod(1)
    a.findEquilibrium()
    busLaneDistance = np.arange(50, 3950, 100)
    busSpeed = []
    carSpeedA = []
    carSpeedB = []
    carSpeedC = []
    carSpeedD = []
    busModeShare = []
    carModeShare = []
    userCosts = []
    operatorCosts = []
    ldCosts = []
    allCosts = []
    initialDistance = a.scenarioData['subNetworkData'].at[2, "Length"]
    for dist in busLaneDistance:
        a.scenarioData['subNetworkData'].at[10, "Length"] = dist
        a.scenarioData['subNetworkData'].at[2, "Length"] = initialDistance - dist
        a.microtypes.updateNetworkData()
        a.findEquilibrium()
        ms = a.getModeSplit(1)

        speeds = pd.DataFrame(a.microtypes.getModeSpeeds())
        busSpeed.append(speeds.loc["B", "bus"])
        carSpeedA.append(speeds.loc["A", "auto"])
        carSpeedB.append(speeds.loc["B", "auto"])
        carSpeedC.append(speeds.loc["C", "auto"])
        carSpeedD.append(speeds.loc["D", "auto"])
        busModeShare.append(ms[a.scenarioData.modeToIdx["bus"]])
        carModeShare.append(ms[a.scenarioData.modeToIdx["auto"]])
        operatorCosts, vectorUserCosts, externalities = a.collectAllCosts()
        ldCosts.append(0.014 * dist)
        x, y = a.plotAllDynamicStats("costs")
        allCosts.append(y.sum().sum())

    plt.scatter(busLaneDistance, busSpeed, marker='<', label="Bus")

    plt.scatter(busLaneDistance, carSpeedA, label="A")
    plt.scatter(busLaneDistance, carSpeedB, label="B")
    plt.scatter(busLaneDistance, carSpeedC, label="C")
    plt.scatter(busLaneDistance, carSpeedD, label="D")
    plt.xlabel("Bus Lane Distance In Microtype B")
    plt.ylabel("Bus Speed In Microtype B")
    plt.legend()
    # plt.xlabel("Bus Lane Distance In Microtype A")
    # plt.ylabel("Bus Speed In Microtype A")
    if not os.path.exists(ROOT_DIR + "/../plots"):
        os.mkdir(ROOT_DIR + "/../plots")
    plt.savefig(ROOT_DIR + "/../plots/buslanevsspeed.png")

    plt.clf()

    plt.scatter(busLaneDistance, busModeShare)
    plt.xlabel("Bus Lane Distance In Microtype B")
    plt.ylabel("Bus Mode Share")
    if not os.path.exists(ROOT_DIR + "/../plots"):
        os.mkdir(ROOT_DIR + "/../plots")
    plt.savefig(ROOT_DIR + "/../plots/buslanevsmodeshare.png")

    plt.clf()
    plt.scatter(busLaneDistance, allCosts)
    # plt.scatter(busLaneDistance, userCosts, label="user")
    # plt.scatter(busLaneDistance, operatorCosts, label="operator")
    # plt.scatter(busLaneDistance, ldCosts, label="lane dedication")
    plt.xlabel("Bus Lane Distance In Microtype B")
    plt.ylabel("Costs")
    plt.legend()
    if not os.path.exists(ROOT_DIR + "/../plots"):
        os.mkdir(ROOT_DIR + "/../plots")
    plt.savefig(ROOT_DIR + "/../plots/buslanevscost.png")

    #    assert busSpeed[-1] / busSpeed[0] > 1.005  # bus lanes speed up bus traffic by a real amount

    a = Model(ROOT_DIR + "/../input-data")
    a.initializeTimePeriod(1)
    a.findEquilibrium()
    headways = np.arange(60, 900, 60)
    busSpeed = []
    carSpeedA = []
    carSpeedB = []
    carSpeedC = []
    carSpeedD = []
    busModeShare = []
    carModeShare = []
    userCosts = []
    operatorCosts = []
    for hw in headways:
        a.scenarioData["modeData"]["bus"].loc["A", "Headway"] = hw
        a.microtypes.updateNetworkData()
        a.findEquilibrium()
        ms = a.getModeSplit(1)
        speeds = pd.DataFrame(a.microtypes.getModeSpeeds())
        busSpeed.append(speeds.loc["B", "bus"])
        carSpeedA.append(speeds.loc["A", "auto"])
        carSpeedB.append(speeds.loc["B", "auto"])
        carSpeedC.append(speeds.loc["C", "auto"])
        carSpeedD.append(speeds.loc["D", "auto"])
        busModeShare.append(ms[a.scenarioData.modeToIdx["bus"]])
        carModeShare.append(ms[a.scenarioData.modeToIdx["auto"]])
        operatorCostDF, vectorUserCosts, externalities = a.collectAllCosts()
        x, y = a.plotAllDynamicStats("costs")
        userCosts.append(y.loc['User'].sum())
        operatorCosts.append(y.loc['Operator'].sum())

    plt.clf()
    plt.scatter(headways, busSpeed, marker='<', label="Bus")
    plt.xlabel("Bus Headway In Microtype A")
    plt.ylabel("Bus Headway In Microtype A")

    plt.scatter(headways, carSpeedA, label="A")
    plt.scatter(headways, carSpeedB, label="B")
    plt.scatter(headways, carSpeedC, label="C")
    plt.scatter(headways, carSpeedD, label="D")
    plt.legend()
    # plt.xlabel("Bus Lane Distance In Microtype A")
    # plt.ylabel("Bus Speed In Microtype A")
    if not os.path.exists(ROOT_DIR + "/../plots"):
        os.mkdir(ROOT_DIR + "/../plots")
    plt.savefig(ROOT_DIR + "/../plots/headwayvsspeed.png")

    plt.clf()

    plt.scatter(headways, busModeShare)
    plt.xlabel("Bus Headway In Microtype A")
    plt.ylabel("Bus Mode Share")
    if not os.path.exists(ROOT_DIR + "/../plots"):
        os.mkdir(ROOT_DIR + "/../plots")
    plt.savefig(ROOT_DIR + "/../plots/headwayvsmodeshare.png")

    plt.clf()
    plt.scatter(headways, userCosts)
    # plt.scatter(headways, operatorCosts)
    plt.xlabel("Bus Headway In Microtype A")
    plt.ylabel("User costs")
    if not os.path.exists(ROOT_DIR + "/../plots"):
        os.mkdir(ROOT_DIR + "/../plots")
    plt.savefig(ROOT_DIR + "/../plots/headwayvscost.png")


test_find_equilibrium()
