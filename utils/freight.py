import pandas as pd
import numpy as np
from utils.data import Data


class FreightMode:
    def __init__(self, name, networks: list, params: pd.DataFrame, microtypeId: str, microtypePopulation: float,
                 demandData: np.ndarray, microtypeSpeed: np.ndarray, fleetSize: np.ndarray):
        self.name = name
        self.params = params
        self._params = params.to_numpy()
        self.modeParamsColumnToIdx = {i: params.index.get_loc(i) for i in params.index}
        self.microtypeId = microtypeId
        self.microtypePopulation = microtypePopulation
        self.microtypeSpeed = microtypeSpeed
        self.microtypeProduction = demandData
        self.microtypeAccumulation = fleetSize

        self._networkSpeed = [n.getModeNetworkSpeed(name) for n in networks]
        self._networkOperatingSpeed = [n.getModeOperatingSpeed(name) for n in networks]
        self._networkBlockedDistance = [n.getModeBlockedDistance(name) for n in networks]
        self._networkAccumulation = [n.getModeAccumulation(name) for n in networks]
        self._networkLength = [n.networkLength for n in networks]
        self.fixedVMT = True
        for n in networks:
            n.addFreightMode(self)

    @property
    def relativeLength(self):
        return self._params[self.modeParamsColumnToIdx["VehicleSize"]]

    @property
    def operatingCostPerHour(self):
        return self._params[self.modeParamsColumnToIdx["OperatingCostPerHour"]]

    def getDemandForVmtPerHour(self):
        return self.microtypeProduction[0]

    def getPortionDedicated(self):
        return 0.0

    def updateFleetSpeed(self):
        self._networkOperatingSpeed[0][0] = self._networkSpeed[0][0]
        self.microtypeSpeed[0] = self._networkSpeed[0][0]

    def assignVmtToNetworks(self):
        # Eventually expand this to handle multiple networks
        productionInVehicleMetersPerSecond = self.microtypeProduction[0] * 1609.34 / 3600
        operatingSpeed = self._networkOperatingSpeed[0][0]
        self._networkAccumulation[0][0] = productionInVehicleMetersPerSecond / operatingSpeed
        self.microtypeAccumulation[0] = self._networkAccumulation[0][0]

    def getOperatorCosts(self):
        return self.microtypeAccumulation[0] * self.operatingCostPerHour

    def updateModeBlockedDistance(self):
        pass
