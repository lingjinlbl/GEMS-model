import pandas as pd
import numpy as np
from utils.data import Data


class FreightMode:
    def __init__(self, name, networks: list, params: pd.DataFrame, microtypeSpeed: np.ndarray,
                 microtypeProduction: np.ndarray, microtypeMixedTrafficDistance: np.ndarray):
        self.name = name
        self.params = params
        self.microtypeSpeed = microtypeSpeed
        self.microtypeProduction = microtypeProduction
        self.microtypeMixedTrafficDistance

        self._networkSpeed = [n.getModeNetworkSpeed(name) for n in networks]
        self._networkOperatingSpeed = [n.getModeOperatingSpeed(name) for n in networks]
        self._networkBlockedDistance = [n.getModeBlockedDistance(name) for n in networks]
        self._networkAccumulation = [n.getModeAccumulation(name) for n in networks]
        self._networkLength = [n.networkLength for n in networks]
