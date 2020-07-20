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
    busLaneDistance = np.arange(0, 5000, 500)
    busSpeed = []
    carSpeed = []
    busModeShare = []
    carModeShare = []
    for dist in busLaneDistance:
        a.scenarioData['subNetworkData'].at[12, "Length"] = dist
        a.scenarioData['subNetworkData'].at[1, "Length"] = 5000 - dist
        a.findEquilibrium()
        ms = a.getModeSplit()
        speeds = pd.DataFrame(a.microtypes.getModeSpeeds())
        busSpeed.append(speeds.loc["bus", "A"])
        carSpeed.append(speeds.loc["auto", "A"])
        busModeShare.append(ms["bus"])
        carModeShare.append(ms["auto"])

    plt.scatter(busLaneDistance, busSpeed)
    plt.xlabel("Bus Lane Distance In Microtype A")
    plt.ylabel("Bus Speed In Microtype A")
    os.mkdir(ROOT_DIR + "/../plots")
    plt.savefig(ROOT_DIR + "/../plots/buslanevsspeed.png")
    assert busSpeed[-1] / busSpeed[0] > 1.005  # bus lanes speed up bus traffic by a real amount
