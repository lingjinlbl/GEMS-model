from math import sqrt, cosh, sinh, cos, sin
from typing import List, Dict

import numba as nb

from utils.supply import TravelDemands
from utils.freight import FreightMode
from utils.passengerModes import *

np.seterr(all='ignore')

mph2mps = 1609.34 / 3600


class NetworkFlowParams:
    def __init__(self, smoothing, free_flow_speed, wave_velocity, jam_density, max_flow, avg_link_length):
        self.lam = smoothing
        self.freeFlowSpeed = free_flow_speed
        self.w = wave_velocity
        self.kappa = jam_density
        self.Q = max_flow
        self.avgLinkLength = avg_link_length


class Costs:
    def __init__(self, per_meter=0.0, per_start=0.0, per_end=0.0, vott_multiplier=1.0):
        self.perMeter = per_meter
        self.perStart = per_start
        self.perEnd = per_end
        self.vottMultiplier = vott_multiplier


class Network:
    def __init__(self, data, characteristics, subNetworkId, diameter=None, microtypeID=None, modeNetworkSpeed=None,
                 modeOperatingSpeed=None, modeAccumulation=None, modeBlockedDistance=None, modeVehicleSize=None,
                 networkLength=None, networkMaxDensity=None, MFD=(), maxInflow=(), modeToIdx=None):
        self.data = data

        self.characteristics = characteristics
        self.charColumnToIdx = {i: characteristics.columns.get_loc(i) for i in characteristics.columns}
        self.dataColumnToIdx = {i: data.columns.get_loc(i) for i in data.columns}
        self.microtypeID = microtypeID
        self.subNetworkId = subNetworkId
        self._iloc = data.index.get_loc(subNetworkId)
        self.__data = data.iloc[self._iloc, :].to_numpy()
        self.type = self.characteristics.iat[self._iloc, self.charColumnToIdx["Type"]]
        self._modes = dict()
        self.dedicated = characteristics.loc[subNetworkId, "Dedicated"]
        self.isJammed = False
        self.__modeToIdx = modeToIdx
        self.modeNetworkSpeed = modeNetworkSpeed
        self.modeOperatingSpeed = modeOperatingSpeed
        self.modeAccumulation = modeAccumulation
        self.modeVehicleSize = modeVehicleSize
        self.networkLength = networkLength
        self.networkMaxDensity = networkMaxDensity
        self.modeBlockedDistance = modeBlockedDistance
        self.__MFD = MFD
        self.__maxInflow = maxInflow
        np.copyto(self.networkLength, self.__data[self.dataColumnToIdx["Length"]])
        if len(self.__MFD) == 0:
            self.MFD = self.defineMFD()
            self.maxInflow = self.defineMaxInflow()
        if diameter is None:
            self.__diameter = 1.0
        else:
            self.__diameter = diameter
        self.modeNetworkSpeed.fill(self.freeFlowSpeed)
        self.modeOperatingSpeed.fill(self.freeFlowSpeed)

    @property
    def MFD(self):
        return self.__MFD[0]

    @MFD.setter
    def MFD(self, value):
        if len(self.__MFD) == 0:
            self.__MFD.append(value)
        else:
            self.__MFD[0] = value

    @property
    def maxInflow(self):
        return self.__maxInflow[0]

    @maxInflow.setter
    def maxInflow(self, value):
        if len(self.__maxInflow) == 0:
            self.__maxInflow.append(value)
        else:
            self.__maxInflow[0] = value

    def runSingleNetworkMFD(self):
        Ntot = (self.modeAccumulation * self.modeVehicleSize).sum()
        Leff = self.L - self.modeBlockedDistance.sum()
        newspeed = self.MFD(Ntot / Leff)
        self.modeNetworkSpeed.fill(newspeed)

    def getModeAccumulation(self, mode):
        return self.modeAccumulation[self.__modeToIdx[mode], None]

    def getAccumulationExcluding(self, mode: str):
        return np.sum(self.modeAccumulation[idx] for m, idx in self.__modeToIdx.items() if m != mode)

    def setModeAccumulation(self, mode, accumulation: float):
        np.copyto(self.modeAccumulation[self.__modeToIdx[mode], None], accumulation)

    def getModeNetworkSpeed(self, mode):
        return self.modeNetworkSpeed[self.__modeToIdx[mode], None]

    def setModeNetworkSpeed(self, mode, speed: float):
        np.copyto(self.modeNetworkSpeed[self.__modeToIdx[mode], None], speed)

    def getModeOperatingSpeed(self, mode):
        return self.modeOperatingSpeed[self.__modeToIdx[mode], None]

    def setModeOperatingSpeed(self, mode, speed: float):
        np.copyto(self.modeOperatingSpeed[self.__modeToIdx[mode], None], speed)

    def getModeBlockedDistance(self, mode):
        return self.modeBlockedDistance[self.__modeToIdx[mode], None]

    def setModeBlockedDistance(self, mode, blockedDistance: float):
        np.copyto(self.modeBlockedDistance[self.__modeToIdx[mode], None], blockedDistance)

    def setModeVehicleSize(self, mode, vehicleSize: float):
        np.copyto(self.modeVehicleSize[self.__modeToIdx[mode], None], vehicleSize)

    def defineMaxInflow(self):
        if self.characteristics.iat[self._iloc, self.charColumnToIdx["Type"]] == "Road":
            densityMax = self.__data[self.dataColumnToIdx["k_jam"]]
            if self.characteristics.iat[self._iloc, self.charColumnToIdx["MFD"]] == "modified-quadratic":
                inflowMax = - self.__data[self.dataColumnToIdx["a"]] * self.__data[self.dataColumnToIdx["b"]] ** 2.0
            else:
                inflowMax = 1.0

            @nb.cfunc("float64(float64)", fastmath=True, parallel=False, cache=True)
            def _maxInflow(density):
                return (densityMax - density) * inflowMax
        else:
            def _maxInflow(density):
                return np.inf
        return _maxInflow

    def defineMFD(self):
        self.networkMaxDensity[0] = self.jamDensity
        if self.characteristics.iat[self._iloc, self.charColumnToIdx["Type"]] == "Road":
            if self.characteristics.iat[self._iloc, self.charColumnToIdx["MFD"]] == "modified-quadratic":
                a = self.__data[self.dataColumnToIdx["a"]]
                criticalDensity = self.__data[self.dataColumnToIdx["b"]]  # b OR k'
                densityMax = self.__data[self.dataColumnToIdx["k_jam"]]  # k_jam

                # (2 a K - 2 a b ) (1 - B/k) -> -a b^2 / (4 k)
                #  - a b^2 / k 2
                # factor = a * ( b - b/2) = a b / 2
                # spd = ab/2 * (1 - b / k ) = ab/2 - ab^2 / 2k

                @nb.cfunc("float64(float64)", fastmath=True, parallel=False, cache=True)
                def _MFD(density):
                    crossoverDensity = densityMax - np.sqrt(densityMax ** 2 - densityMax * criticalDensity)
                    # dm (1 - sqrt(1 - crit/dm))
                    if density <= 0:
                        return -a * criticalDensity
                    elif density < crossoverDensity:
                        return a * (density - criticalDensity)
                    else:
                        factor = a * (2 * crossoverDensity - criticalDensity)
                        return max(factor * (density - densityMax) / density, 0.1)

            elif self.characteristics.iat[self._iloc, self.charColumnToIdx["MFD"]] == "loder":
                vMax = self.__data[self.dataColumnToIdx["vMax"]]
                densityMax = self.__data[self.dataColumnToIdx["k_jam"]]
                capacityFlow = self.__data[self.dataColumnToIdx["capacityFlow"]]
                smoothingFactor = self.__data[self.dataColumnToIdx["smoothingFactor"]]
                waveSpeed = self.__data[self.dataColumnToIdx["waveSpeed"]]

                @nb.cfunc("float64(float64)", fastmath=True, parallel=False, cache=True)
                def _MFD(density):
                    speedMin = 0.0222
                    if density == 0:
                        return vMax
                    elif density > densityMax * 0.99:
                        return speedMin
                    else:
                        speedExp = smoothingFactor / density * np.log(
                            np.exp(- vMax * density / smoothingFactor) + np.exp(
                                -capacityFlow / smoothingFactor) + np.exp(
                                - (density - densityMax) * waveSpeed / smoothingFactor))
                        speedLinear = vMax * (1. - density / densityMax)
                        if speedExp > speedLinear:
                            return speedLinear
                        else:
                            return speedExp

            elif self.characteristics.iat[self._iloc, self.charColumnToIdx["MFD"]] == "quadratic":
                vMax = self.__data[self.dataColumnToIdx["vMax"]]
                densityMax = self.__data[self.dataColumnToIdx["k_jam"]]

                @nb.cfunc("float64(float64)", fastmath=True, parallel=False, cache=True)
                def _MFD(density):
                    return max(vMax * (1. - density / densityMax), 0.0333)

            elif self.characteristics.iat[self._iloc, self.charColumnToIdx["MFD"]] == "bottleneck":
                vMax = self.__data[self.dataColumnToIdx["vMax"]]
                capacityFlow = self.__data[self.dataColumnToIdx["capacityFlow"]]

                @nb.cfunc("float64(float64)", fastmath=True, parallel=False, cache=True)
                def _MFD(density):
                    if density > (capacityFlow / vMax):
                        return capacityFlow / density
                    else:
                        return vMax

            elif self.characteristics.iat[self._iloc, self.charColumnToIdx["MFD"]] == "rural":
                a = self.__data[self.dataColumnToIdx["a"]]
                b = self.__data[self.dataColumnToIdx["b"]]

                @nb.cfunc("float64(float64)", fastmath=True, parallel=False, cache=True)
                def _MFD(density):
                    if density < (b / 2):
                        out = max(a * (density - b), 0.01)
                        return out
                    else:
                        out = max(-a * b ** 2 / (4 * density), 0.01)
                        return out

            else:
                vMax = self.__data[self.dataColumnToIdx["vMax"]]

                @nb.cfunc("float64(float64)", fastmath=True, parallel=False, cache=True)
                def _MFD(_):
                    return vMax
        else:
            vMax = self.__data[self.dataColumnToIdx["vMax"]]

            def _MFD(_):
                return vMax

        return _MFD

    def updateNetworkData(self):  # CONSOLIDATE
        np.copyto(self.__data, self.data.iloc[self._iloc, :].to_numpy())
        self.MFD = self.defineMFD()

    # @property
    # def type(self):
    #     return self.characteristics.iat[self._idx, self.charColumnToIdx["Type"]]

    @property
    def base_speed(self):
        return self.modeNetworkSpeed[self.__modeToIdx['auto']]

    @property
    def autoSpeed(self):
        return self.modeNetworkSpeed[self.__modeToIdx['auto']]

    @autoSpeed.setter
    def autoSpeed(self, newSpeed):
        self.modeNetworkSpeed[self.__modeToIdx['auto']] = newSpeed

    @base_speed.setter
    def base_speed(self, spd):
        self.modeNetworkSpeed[self.__modeToIdx['auto']] = spd

    @property
    def avgLinkLength(self):
        return self.__data[self.dataColumnToIdx["avgLinkLength"]]

    @property
    def freeFlowSpeed(self):
        return self.MFD(0.0)

    @property
    def jamDensity(self):
        densityMax = self.__data[self.dataColumnToIdx["k_jam"]]
        if np.isnan(densityMax) | (densityMax <= 0.0):
            return self.__data[self.dataColumnToIdx["capacityFlow"]] / self.__data[self.dataColumnToIdx["vMax"]] * 4.
        else:
            return densityMax

    @property
    def L(self):
        return self.networkLength[0]
        # return self.__data[self.dataColumnToIdx["Length"]]

    @property
    def diameter(self):
        return self.__diameter

    @property
    def modesAllowed(self):
        return self.characteristics['ModesAllowed'].iloc[self._iloc]

    def __str__(self):
        return str(self._modes) + '__' + str(self.modeAccumulation)

    def __contains__(self, mode):
        return mode in self._modes

    def updateScenarioInputs(self):
        np.copyto(self.__data, self.data.iloc[self._iloc, :].to_numpy())

    def resetAll(self):
        # self.L_blocked = dict()
        self._modes = dict()
        self.base_speed = self.freeFlowSpeed
        self.isJammed = False

    def resetSpeeds(self):
        # self.base_speed = self.freeFlowSpeed
        self.isJammed = False
        # for key in self.L_blocked.keys():
        #     self.L_blocked[key] = 0.0

    # def resetModes(self):
    #     for mode in self._modes.values():
    #         # self.N_eq[mode.name] = mode.getN(self) * mode.params.relativeLength
    #         self._VMT[mode] = mode._VMT[self]
    #         self._N_eff[mode] = mode._N_eff[self]
    #         self.L_blocked[mode.name] = mode.getBlockedDistance(self)
    #     self.isJammed = False
    #     self.base_speed = self.freeFlowSpeed
    # mode.reset()

    def setVMT(self, mode: str, VMT: float):
        self._VMT[mode] = VMT

    def setN(self, mode: str, N: float):
        self._N_eff[mode] = N

    # def updateBaseSpeed(self, override=False):
    #     # out = self.NEF(overrideMatrix=override)
    #     if self.dedicated:
    #         self.base_speed = self.NEF(overrideMatrix=override)

    def getSpeedFromMFD(self, N):
        L_tot = self.L - self.getBlockedDistance()
        N_0 = self.jamDensity * L_tot
        return self.freeFlowSpeed * (1. - N / N_0)

    def NEF2(self) -> float:
        if self.type == "Road":
            return self._V_mean
        else:
            return self.freeFlowSpeed

    def NEF(self, Q=None, modeIgnored=None, overrideMatrix=False) -> float:
        if self.type == 'Road':
            if 'auto' in self.getModeNames() and not overrideMatrix:
                # print("THIS WILL BREAK THINGS")
                return self.modeNetworkSpeed[self.__modeToIdx['auto']]
            else:
                if Q is None:
                    Qtot = sum([VMT for VMT in self._VMT.values()]) * mph2mps
                else:
                    Qtot = Q
                    for mode, Qmode in self._VMT.items():
                        if mode != modeIgnored:
                            Qtot += Qmode * mph2mps
                if Qtot == 0:
                    return self.freeFlowSpeed

                L_tot = self.L - self.getBlockedDistance()
                L_0 = 10 * 1609.34  # TODO: Get average distance, don't hardcode
                t = 3 * 3600.  # TODO: Add timestep duration in seconds
                N_0 = self.jamDensity * L_tot
                V_0 = self.freeFlowSpeed
                N_init = self._N_init
                if N_0 ** 2. / 4. >= N_0 * Qtot / V_0:
                    # Stable state
                    A = sqrt(N_0 ** 2. / 4. - N_0 * Qtot / V_0)
                    var = A * V_0 * t / (N_0 * L_0)
                    N_final = N_0 / 2 - A * ((N_0 / 2 - N_init) * cosh(var) + A * sinh(var)) / (
                            (N_0 / 2 - N_init) * sinh(var) + A * cosh(var))
                    V_init = self.getSpeedFromMFD(N_init)
                    V_final = self.getSpeedFromMFD(N_final)
                    V_steadyState = self.getSpeedFromMFD(N_0 / 2 - A)
                else:
                    Aprime = sqrt(N_0 * Qtot / V_0 - N_0 ** 2. / 4.)
                    var = Aprime * V_0 * t / (N_0 * L_0)
                    N_final = N_0 / 2 - Aprime * ((N_0 / 2 - N_init) * cos(var) + Aprime * sin(var)) / (
                            (N_0 / 2 - N_init) * sin(var) + Aprime * cos(var))
                    V_init = self.getSpeedFromMFD(N_init)
                    V_final = self.getSpeedFromMFD(N_final)
                    V_steadyState = 0
                if overrideMatrix:
                    self.base_speed = max([0.1, (V_init + V_final) / 2.0])
                return max([2.0, (V_init + V_final) / 2.0])  # TODO: Actually take the integral
        else:
            return self.freeFlowSpeed

    def getBaseSpeed(self):
        if self.base_speed > 0.01:
            return self.base_speed
        else:
            return 0.01

    # def updateBlockedDistance(self):
    #     for mode in self._modes.values():
    #         mode.updateModeBlockedDistance()

    def containsMode(self, mode: str) -> bool:
        return mode in self._modes.keys()

    def getBlockedDistance(self) -> float:
        return self.modeBlockedDistance.sum()

    def addPassengerMode(self, mode: Mode):
        self.setModeVehicleSize(mode.name, mode.relativeLength)
        self._modes[mode.name] = mode

    def addFreightMode(self, mode: FreightMode):
        self.setModeVehicleSize(mode.name, mode.relativeLength)
        self._modes[mode.name] = mode

    def getModeNames(self) -> list:
        return list(self._modes.keys())

    def getModeValues(self) -> list:
        return list(self._modes.values())

    def getVMT(self, mode):
        return self.modeAccumulation(mode) * self.modeNetworkSpeed(mode)


class NetworkCollection:
    def __init__(self, networksAndModes, modeToModeData, microtypeID, microtypePopulation, demandData, speedData,
                 microtypeCosts, fleetSize, freightProduction, accessDistance, demandDataTypeToIdx, modeToIdx,
                 freightModeToIdx, diToIdx, verbose=False):
        self._networks = dict()
        self.modeToNetwork = dict()
        self.__passengerModes = dict()
        self.__freightModes = dict()
        self.__microtypePopulation = microtypePopulation
        self.__demandData = demandData
        self._speedData = speedData
        self.__microtypeCosts = microtypeCosts
        self.__fleetSize = fleetSize
        self.__freightProduction = freightProduction
        self.__accessDistance = accessDistance
        self.__demandDataTypeToIdx = demandDataTypeToIdx
        self.__modeToIdx = modeToIdx
        self.__freightModeToIdx = freightModeToIdx
        self.__diToIdx = diToIdx

        # self.demands = TravelDemands([])
        self.verbose = verbose

        if isinstance(networksAndModes, Dict) and isinstance(modeToModeData, Dict):
            self.populateNetworksAndModes(networksAndModes, modeToModeData, microtypeID)

        # self.resetModes()

    @property
    def microtypePopulation(self):
        return self.__microtypePopulation

    def subNetworkIDs(self):
        return list(self._networks.keys())

    def passengerModes(self):
        return self.__passengerModes

    def freightModes(self):
        return self.__freightModes

    def getMode(self, mode):
        if mode in self.__passengerModes:
            return self.__passengerModes[mode]
        else:
            return self.__freightModes[mode]

    def fixedVMT(self, mode):
        if mode in self.__passengerModes:
            return self.__passengerModes[mode].fixedVMT
        elif mode in self.__freightModes:
            return self.__freightModes[mode].fixedVMT
        else:
            return False

    def getModeVMT(self, mode):
        return self.getMode(mode).getDemandForVmtPerHour()

    def updateNetworkData(self):
        for n in self._networks.values():
            n.updateNetworkData()

    def populateNetworksAndModes(self, networksAndModes, modeToModeData, microtypeID):
        # modeToNetwork = dict()
        if isinstance(networksAndModes, Dict):
            for (network, modeNames) in networksAndModes.items():
                self._networks[network.subNetworkId] = network
                for modeName in modeNames:
                    if modeName in self.modeToNetwork:
                        self.modeToNetwork[modeName].append(network)
                    else:
                        self.modeToNetwork[modeName] = [network]

        else:
            print('Bad NetworkCollection Input')
        for (modeName, networks) in self.modeToNetwork.items():
            assert (isinstance(modeName, str))
            assert (isinstance(networks, List))
            params = modeToModeData[modeName]

            travelDemandData = self.__demandData[self.__modeToIdx[modeName], :]
            speedData = self._speedData[self.__modeToIdx[modeName], None]
            microtypeCosts = self.__microtypeCosts[:, self.__modeToIdx[modeName], :]
            fleetSize = self.__fleetSize[self.__modeToIdx[modeName], None]
            accessDistance = self.__accessDistance[self.__modeToIdx[modeName], None]

            if modeName == "bus":
                self._speedData[self.__modeToIdx[modeName]] = networks[0].autoSpeed
                mode = BusMode(networks, params, microtypeID, self.microtypePopulation,
                               travelDemandData=travelDemandData, speedData=speedData, microtypeCosts=microtypeCosts,
                               fleetSize=fleetSize, accessDistance=accessDistance, diToIdx=self.__diToIdx)
                self.__passengerModes["bus"] = mode
            elif modeName == "auto":
                self._speedData[self.__modeToIdx[modeName]] = networks[0].autoSpeed
                mode = AutoMode(networks, params, microtypeID, self.microtypePopulation,
                                travelDemandData=travelDemandData, speedData=speedData,
                                microtypeCosts=microtypeCosts, fleetSize=fleetSize, accessDistance=accessDistance,
                                diToIdx=self.__diToIdx)
                self.__passengerModes["auto"] = mode
            elif modeName == "walk":
                self._speedData[self.__modeToIdx[modeName]] = params.loc[microtypeID, 'SpeedInMetersPerSecond']
                mode = WalkMode(networks, params, microtypeID, self.microtypePopulation,
                                travelDemandData=travelDemandData, speedData=speedData, microtypeCosts=microtypeCosts,
                                fleetSize=fleetSize, accessDistance=accessDistance, diToIdx=self.__diToIdx)
                self.__passengerModes["walk"] = mode
            elif modeName == "bike":
                self._speedData[self.__modeToIdx[modeName]] = params.loc[microtypeID, 'SpeedInMetersPerSecond']
                mode = BikeMode(networks, params, microtypeID, self.microtypePopulation,
                                travelDemandData=travelDemandData, speedData=speedData,
                                microtypeCosts=microtypeCosts, fleetSize=fleetSize, accessDistance=accessDistance,
                                diToIdx=self.__diToIdx)
                self.__passengerModes["bike"] = mode
            elif modeName == "rail":
                self._speedData[self.__modeToIdx[modeName]] = params.loc[microtypeID, 'SpeedInMetersPerSecond']
                mode = RailMode(networks, params, microtypeID, self.microtypePopulation,
                                travelDemandData=travelDemandData, speedData=speedData,
                                microtypeCosts=microtypeCosts, fleetSize=fleetSize, accessDistance=accessDistance,
                                diToIdx=self.__diToIdx)
                self.__passengerModes["rail"] = mode
            elif modeName.startswith("freight"):
                self._speedData[self.__modeToIdx[modeName]] = networks[0].autoSpeed
                freightProduction = self.__freightProduction[self.__freightModeToIdx[modeName], None]
                mode = FreightMode(modeName, networks, params, microtypeID, self.microtypePopulation,
                                   travelDemandData[self.__demandDataTypeToIdx['vehicleDistance'], None], speedData,
                                   fleetSize)
                self.__freightModes[modeName] = mode
            else:
                print("BAD!")
                Mode(networks, params, microtypeID, "bad")
        # for modeName, mode in self.__passengerModes.items():
        #     self.demands[modeName] = mode.travelDemand

    def updateModeData(self):
        for m in self.__passengerModes.values():
            m.updateScenarioInputs()
            m.updateModeBlockedDistance()

    def isJammed(self):
        return np.any([n.isJammed for n in self._networks])

    def resetModes(self):  # NOTE: Took some of this out, NOV 2021
        for n in self._networks.values():
            n.isJammed = False

    # @profile
    def updateModes(self, nIters: int = 1):
        for m in self.passengerModes().values():  # uniqueModes:
            # replace this with assign accumulation to networks
            m.assignVmtToNetworks()
            m.updateModeBlockedDistance()
            m.updateRouteAveragedSpeed()
        for m in self.freightModes().values():
            m.updateFleetSpeed()
            m.assignVmtToNetworks()
            m.updateModeBlockedDistance()

    def __getitem__(self, item):
        return self._networks[item]

    def __str__(self):
        return str([n.base_speed for n in self._networks])

    def __iter__(self):
        return iter(self._networks.values())

    def __contains__(self, mode):
        return mode in self.modeToNetwork

    def getModeNames(self) -> list:
        return list(self.modeToNetwork.keys())

    def getModeSpeeds(self) -> np.array:
        # TODO: This can get slow
        return self._speedData

    def getModeOperatingCosts(self):
        out = np.zeros(len(self.__modeToIdx))
        for name, mode in self.passengerModes().items():
            out[self.__modeToIdx[name]] = mode.getOperatorCosts()
        return out

    def getFreightModeOperatingCosts(self):
        out = np.zeros(len(self.__modeToIdx))
        for name, mode in self.freightModes().items():
            out[self.__modeToIdx[name]] = mode.getOperatorCosts()
        return out

    def iterModes(self):
        return iter(self.__passengerModes)
