import pandas as pd
import matplotlib.pyplot as plt
import os
from model import Model
import pytest
import numpy as np


def test_find_equilibrium():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    a = Model(ROOT_DIR + "/../input-data")
    a.initializeTimePeriod("AM-Peak")
    a.findEquilibrium()
    busLaneDistance = np.arange(500, 4500, 500)
    busSpeed = []
    carSpeedA = []
    carSpeedB = []
    carSpeedC = []
    carSpeedD = []
    busModeShare = []
    carModeShare = []
    for dist in busLaneDistance:
        a.scenarioData['subNetworkData'].at[13, "Length"] = dist
        a.scenarioData['subNetworkData'].at[2, "Length"] = 5000 - dist
        a.findEquilibrium()
        ms = a.getModeSplit()
        speeds = pd.DataFrame(a.microtypes.getModeSpeeds())
        busSpeed.append(speeds.loc["bus", "A"])
        carSpeedA.append(speeds.loc["auto", "A"])
        carSpeedB.append(speeds.loc["auto", "B"])
        carSpeedC.append(speeds.loc["auto", "C"])
        carSpeedD.append(speeds.loc["auto", "D"])
        busModeShare.append(ms["bus"])
        carModeShare.append(ms["auto"])

    plt.scatter(busLaneDistance, busSpeed)
    plt.xlabel("Bus Lane Distance In Microtype A")
    plt.ylabel("Bus Speed In Microtype A")


    plt.scatter(busLaneDistance, carSpeedA)
    plt.scatter(busLaneDistance, carSpeedB)
    plt.scatter(busLaneDistance, carSpeedC)
    plt.scatter(busLaneDistance, carSpeedD)
    # plt.xlabel("Bus Lane Distance In Microtype A")
    # plt.ylabel("Bus Speed In Microtype A")
    if not os.path.exists(ROOT_DIR + "/../plots"):
        os.mkdir(ROOT_DIR + "/../plots")
    plt.savefig(ROOT_DIR + "/../plots/buslanevsspeed.png")

    plt.clf()

    plt.scatter(busLaneDistance, busModeShare)
    plt.xlabel("Bus Lane Distance In Microtype A")
    plt.ylabel("Bus Mode Share")
    if not os.path.exists(ROOT_DIR + "/../plots"):
        os.mkdir(ROOT_DIR + "/../plots")
    plt.savefig(ROOT_DIR + "/../plots/buslanevsmodeshare.png")
    assert busSpeed[-1] / busSpeed[0] > 1.005  # bus lanes speed up bus traffic by a real amount
