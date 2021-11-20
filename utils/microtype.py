#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import OrderedDict

import numpy as np
import pandas as pd
from numba import njit

from .OD import TransitionMatrix, Allocation
from .choiceCharacteristics import ChoiceCharacteristics
from .network import Network, NetworkCollection, Costs, TotalOperatorCosts, CollectedNetworkStateData


class CollectedTotalOperatorCosts:
    def __init__(self):
        self.__costs = dict()
        self.total = 0.

    def __setitem__(self, key: str, value: TotalOperatorCosts):
        self.__costs[key] = value
        self.updateTotals(value)

    def __getitem__(self, item: str) -> TotalOperatorCosts:
        return self.__costs[item]

    def updateTotals(self, value: TotalOperatorCosts):
        for mode, cost in value:
            self.total += cost

    def __mul__(self, other):
        out = CollectedTotalOperatorCosts()
        for mode in self.__costs.keys():
            out[mode] = self[mode] * other
        return out

    def __add__(self, other):
        out = CollectedTotalOperatorCosts()
        for mode in other.__costs.keys():
            if mode in self.__costs:
                out[mode] = self[mode] + other[mode]
            else:
                out[mode] = other[mode]
        return out

    def __iadd__(self, other):
        for mode in other.__costs.keys():
            if mode in self.__costs:
                self[mode] = self[mode] + other[mode]
            else:
                self[mode] = other[mode]
        return self

    def toDataFrame(self):
        return pd.concat([val.toDataFrame([key]) for key, val in self.__costs.items()])

    def __str__(self):
        return str(self.toDataFrame())


class Microtype:
    def __init__(self, microtypeID: str, networks: NetworkCollection, paramToIdx: dict, costs=None):
        self.microtypeID = microtypeID
        if costs is None:
            costs = dict()
        self.mode_names = set(networks.getModeNames())
        self.networks = networks
        self.updateModeCosts(costs)
        self.paramToIdx = paramToIdx

    def __contains__(self, item):
        return item in self.mode_names

    def updateModeCosts(self, costs):
        for (mode, modeCosts) in costs.items():
            assert (isinstance(mode, str) and isinstance(modeCosts, Costs))
            self.networks.modes[mode].costs = modeCosts

    def updateNetworkSpeeds(self, nIters=None):
        self.networks.updateModes(nIters)

    def getModeSpeeds(self) -> dict:
        return {mode: self.getModeSpeed(mode) for mode in self.mode_names}

    def getModeSpeed(self, mode) -> float:
        return self.networks.modes[mode].getSpeed()

    def getModeFlow(self, mode) -> float:
        return self.networks.demands.getRateOfPMT(mode)

    def getModeDemandForPMT(self, mode):
        return self.networks.demands.getRateOfPMT(mode)

    def addModeStarts(self, mode, demand):
        self.networks.demands.addModeStarts(mode, demand)

    def addModeEnds(self, mode, demand):
        self.networks.demands.addModeEnds(mode, demand)

    def resetDemand(self):
        self.networks.resetModes()
        self.networks.demands.resetDemand()

    def getModeStartAndEndRate(self, mode: str) -> (float, float):
        return self.networks.demands.getStartRate(mode), self.networks.demands.getStartRate(mode)

    def getModeStartRate(self, mode: str) -> float:
        return self.networks.demands.getStartRate(mode)

    def getModeMeanDistance(self, mode: str):
        return self.networks.demands.getAverageDistance(mode)

    def addStartTimeCostWait(self, mode: str, cc: np.ndarray):  # Could eventually be vectorized
        if mode in self:
            cc[self.paramToIdx['cost']] += self.networks.modes[mode].perStart
            if mode in ['bus', 'rail']:
                cc[self.paramToIdx['wait_time']] += self.networks.modes[
                                                        'bus'].headwayInSec / 3600. / 4.  # TODO: Something better than average of start and end
                cc[self.paramToIdx['access_time']] += self.networks.modes[
                                                          mode].getAccessDistance() / 1.5 / 3600.0  # TODO: Switch back to self.networks.modes['walk'].speedInMetersPerSecond

    def addEndTimeCostWait(self, mode: str, cc: np.ndarray):
        if mode in self:
            cc[self.paramToIdx['cost']] += self.networks.modes[mode].perEnd
            if mode == 'bus':
                cc[self.paramToIdx['wait_time']] += self.networks.modes['bus'].headwayInSec / 3600. / 4.
                cc[self.paramToIdx['access_time']] += self.networks.modes[mode].getAccessDistance() * \
                                                      self.networks.modes[
                                                          'walk'].speedInMetersPerSecond / 3600.0


def getFlows(self):
    return [mode.getPassengerFlow() for mode in self.networks.modes.values()]


def getSpeeds(self):
    return [mode.getSpeed() for mode in self.networks.modes.values()]


def getDemandsForPMT(self):
    return [mode.getPassengerFlow() for mode in
            self.networks.modes.values()]


def __str__(self):
    return 'Demand: ' + str(self.getFlows()) + ' , Speed: ' + str(self.getSpeeds())


class MicrotypeCollection:
    def __init__(self, scenarioData, supplyData):
        self.__timeStepInSeconds = scenarioData.timeStepInSeconds
        self.__microtypes = dict()
        self.__scenarioData = scenarioData
        self.modeData = scenarioData["modeData"]
        self.transitionMatrix = None
        self.collectedNetworkStateData = CollectedNetworkStateData()
        self.__modeToMicrotype = dict()
        self.__networkIdToIdx = dict()
        self.__numpyDemand = supplyData['demandData']
        self.__numpyMicrotypeSpeed = supplyData['microtypeSpeed']
        self.__numpyMicrotypeMixedTrafficDistance = supplyData['microtypeMixedTrafficDistance']
        self.__diameters = np.ndarray([0])
        self.__numpyNetworkAccumulation = supplyData['subNetworkAccumulation']
        self.__numpyNetworkLength = supplyData['subNetworkLength']
        self.__numpyNetworkVehicleSize = supplyData['subNetworkVehicleSize']
        self.__numpyNetworkSpeed = supplyData['subNetworkAverageSpeed']
        self.__numpyNetworkOperatingSpeed = supplyData['subNetworkOperatingSpeed']
        self.__numpyNetworkBlockedDistance = supplyData['subNetworkBlockedDistance']
        self.__numpyInstantaneousSpeed = supplyData['subNetworkInstantaneousSpeed']
        self.__numpyInstantaneousAccumulation = supplyData['subNetworkInstantaneousAutoAccumulation']
        self.__transitionMatrixNetworkIdx = supplyData['transitionMatrixNetworkIdx']
        self.__nonAutoModes = supplyData['nonAutoModes']
        self.__nInit = supplyData['subNetworkPreviousAutoAccumulation']

    @property
    def nonAutoModes(self):
        return self.__nonAutoModes

    @property
    def autoThroughDistance(self):
        return self.__numpyDemand[:, self.modeToIdx['auto'], -1]

    @property
    def diToIdx(self):
        return self.__scenarioData.diToIdx

    @property
    def odiToIdx(self):
        return self.__scenarioData.odiToIdx

    @property
    def modeToIdx(self):
        return self.__scenarioData.modeToIdx

    @property
    def paramToIdx(self):
        return self.__scenarioData.paramToIdx

    @property
    def dataToIdx(self):
        return self.__scenarioData.dataToIdx

    @property
    def microtypeIdToIdx(self):
        return self.__scenarioData.microtypeIdToIdx

    @property
    def diameters(self):
        return self.__diameters

    @property
    def numpySpeed(self):
        return self.__numpyMicrotypeSpeed

    @property
    def numpyMixedTrafficDistance(self):
        return self.__numpyMicrotypeMixedTrafficDistance

    @property
    def passengerDistanceByMode(self):
        return self.__numpyDemand[:, :, -2]

    @property
    def vehicleDistanceByMode(self):
        return self.__numpyDemand[:, :, -1]

    @property
    def tripStartRateByMode(self):
        return self.__numpyDemand[:, :, 0]

    @property
    def tripEndRateByMode(self):
        return self.__numpyDemand[:, :, 1]

    def dataByModeDataFrame(self):
        out = {'TripStartsPerHour': self.tripStartRateByMode.flatten(),
               'TripEndsPerHour': self.tripEndRateByMode.flatten(),
               'PassengerDistancePerHour': self.passengerDistanceByMode.flatten(),
               'VehicleDistancePerHour': self.vehicleDistanceByMode.flatten(),
               'Speed': self.__numpyMicrotypeSpeed.flatten()}
        return pd.DataFrame(out,
                            index=pd.MultiIndex.from_product([self.microtypeIdToIdx.keys(), self.modeToIdx.keys()]))

    def tripStartRateByModeDataFrame(self):
        return pd.DataFrame(self.tripStartRateByMode.flatten(),
                            index=pd.MultiIndex.from_product([self.microtypeIdToIdx.keys(), self.modeToIdx.keys()]))

    def tripEndRateByModeDataFrame(self):
        return pd.DataFrame(self.tripEndRateByMode.flatten(),
                            index=pd.MultiIndex.from_product([self.microtypeIdToIdx.keys(), self.modeToIdx.keys()]))

    def updateNumpyDemand(self, data):
        np.copyto(self.__numpyDemand, data)

    def updateNetworkData(self):
        for m in self.__microtypes.values():
            # assert isinstance(m, Microtype)
            m.networks.updateNetworkData()
            m.networks.updateModeData()
            for _, n in m.networks:
                # assert isinstance(n, Network)
                n.updateScenarioInputs()

    def recompileMFDs(self):
        for m in self.__microtypes.values():
            for modes, n in m.networks:
                n.recompileMFD()

    def __setitem__(self, key: str, value: Microtype):
        self.__microtypes[key] = value

    def __getitem__(self, item: str) -> Microtype:
        return self.__microtypes[item]

    def __contains__(self, item):
        return item in self.__microtypes

    def __len__(self):
        return len(self.__microtypes)

    # def getAllStartCosts(self, microtypeToIdx: dict, characteristicToIdx: dict) -> np.ndarray:
    #     out = np.ndarray((len(microtypeToIdx), len(characteristicToIdx)))
    #
    #     return out

    def microtypeNames(self):
        return list(self.__microtypes.keys())

    def getModeStartRatePerSecond(self, mode):
        return np.array([microtype.getModeStartRate(mode) / 3600. for mID, microtype in self])

    def importMicrotypes(self, override=False):
        # uniqueMicrotypes = subNetworkData["MicrotypeID"].unique()

        subNetworkData = self.__scenarioData["subNetworkData"]
        subNetworkCharacteristics = self.__scenarioData["subNetworkDataFull"]
        modeToSubNetworkData = self.__scenarioData["modeToSubNetworkData"]
        microtypeData = self.__scenarioData["microtypeIDs"]
        self.__networkIdToIdx = {networkId: idx for idx, networkId in enumerate(subNetworkCharacteristics.index)}
        self.__diameters = np.zeros(len(microtypeData), dtype=float)
        for microtypeId, idx in self.microtypeIdToIdx.items():
            self.__diameters[idx] = microtypeData.loc[microtypeId, 'DiameterInMiles'] * 1609.34

        self.transitionMatrix = TransitionMatrix(self.microtypeIdToIdx,
                                                 diameters=self.__diameters)

        if len(self.__microtypes) == 0:
            self.__modeToMicrotype = dict()

        collectMatrixIds = (sum(self.__transitionMatrixNetworkIdx) == 0)
        for microtypeID, diameter in microtypeData.itertuples(index=False):
            if (microtypeID in self) & ~override:
                self[microtypeID].resetDemand()
            else:
                subNetworkToModes = OrderedDict()
                modeToModeData = OrderedDict()
                allModes = set()
                for subNetworkId in subNetworkCharacteristics.loc[
                    subNetworkCharacteristics["MicrotypeID"] == microtypeID].index:
                    joined = modeToSubNetworkData.loc[
                        modeToSubNetworkData['SubnetworkID'] == subNetworkId]
                    subNetwork = Network(subNetworkData, subNetworkCharacteristics, subNetworkId, diameter, microtypeID,
                                         self.__numpyNetworkSpeed[self.__networkIdToIdx[subNetworkId], :],
                                         self.__numpyNetworkOperatingSpeed[self.__networkIdToIdx[subNetworkId], :],
                                         self.__numpyNetworkAccumulation[self.__networkIdToIdx[subNetworkId], :],
                                         self.__numpyNetworkBlockedDistance[self.__networkIdToIdx[subNetworkId], :],
                                         self.__numpyNetworkVehicleSize[self.__networkIdToIdx[subNetworkId], :],
                                         self.__numpyNetworkLength[self.__networkIdToIdx[subNetworkId], :],
                                         self.modeToIdx)
                    if collectMatrixIds:
                        if 'auto' in subNetwork.modesAllowed.lower():  # Simple fix for now while we just have 1 auto network per microtype
                            self.__transitionMatrixNetworkIdx[self.__networkIdToIdx[subNetworkId]] = True

                    for n in joined.itertuples():
                        subNetworkToModes.setdefault(subNetwork, []).append(n.ModeTypeID.lower())
                        allModes.add(n.ModeTypeID.lower())
                        self.__modeToMicrotype.setdefault(n.ModeTypeID.lower(), set()).add(microtypeID)
                for mode in allModes:
                    modeToModeData[mode] = self.modeData[mode]
                networkCollection = NetworkCollection(subNetworkToModes, modeToModeData, microtypeID,
                                                      self.__numpyDemand[self.microtypeIdToIdx[microtypeID], :, :],
                                                      self.__numpyMicrotypeSpeed[self.microtypeIdToIdx[microtypeID], :],
                                                      self.dataToIdx, self.modeToIdx)
                self[microtypeID] = Microtype(microtypeID, networkCollection, self.paramToIdx)
                self.collectedNetworkStateData.addMicrotype(self[microtypeID])

                # print("|  Loaded ",
                #       len(subNetworkCharacteristics.loc[subNetworkCharacteristics["MicrotypeID"] == microtypeID].index),
                #       " subNetworks in microtype ", microtypeID)

    def updateDedicatedDistance(self):
        for microtypeID, microtype in self:
            idx = self.transitionMatrix.idx(microtypeID)
            for mode in microtype.mode_names:
                portionDedicated = microtype.networks.modes[mode].getPortionDedicated()
                distanceMixed = (1. - portionDedicated)  # microtype.getModeDemandForPMT(mode) *
                self.__numpyMicrotypeMixedTrafficDistance[idx, self.modeToIdx[mode]] = distanceMixed

    def transitionMatrixMFD(self, durationInHours, collectedNetworkStateData=None, tripStartRate=None):
        if collectedNetworkStateData is None:
            collectedNetworkStateData = self.collectedNetworkStateData

        if tripStartRate is None:
            tripStartRate = self.getModeStartRatePerSecond("auto")

        characteristicL = np.zeros((len(self)), dtype=float)
        L_eff = np.zeros((len(self)), dtype=float)
        # n_init = np.zeros((len(self)), dtype=float)
        speedFunctions = [None] * len(self)
        for microtypeID, microtype in self:
            idx = self.transitionMatrix.idx(microtypeID)
            for modes, autoNetwork in microtype.networks:
                if "auto" in autoNetwork:
                    networkStateData = collectedNetworkStateData[(microtypeID, modes)]
                    characteristicL[idx] += autoNetwork.diameter * 1609.34
                    # n_init[idx] = networkStateData.initialAccumulation
                    speedFunctions[idx] = autoNetwork.MFD
        n_init = self.__nInit[self.__transitionMatrixNetworkIdx]
        L_blocked = self.__numpyNetworkBlockedDistance[self.__transitionMatrixNetworkIdx, :].sum(axis=1)
        L_eff = self.__numpyNetworkLength[self.__transitionMatrixNetworkIdx, 0] - L_blocked
        n_other = (self.__numpyNetworkAccumulation[self.__transitionMatrixNetworkIdx, :][:,
                   self.nonAutoModes] * self.__numpyNetworkVehicleSize[self.__transitionMatrixNetworkIdx, :][:,
                                        self.nonAutoModes]).sum(axis=1)
        dt = self.__timeStepInSeconds

        ts = np.arange(0, durationInHours * 3600., dt)
        ns = np.zeros((len(self), np.size(ts)), dtype=float)
        ns = doMatrixCalcs(ns, n_init, self.transitionMatrix.matrix.values, tripStartRate, characteristicL, L_eff,
                           n_other, dt, speedFunctions)
        if np.any(np.isnan(ns)):
            print('hmmmm')
        vs = vectorV(ns, n_other, L_eff, speedFunctions)

        # inflows = vs.copy()
        # outflows = vs.copy()
        # flowMats = np.zeros((len(self), len(self), np.size(ts)), dtype=float)

        averageSpeeds = np.sum(ns, axis=1) / np.sum(ns / vs, axis=1)

        # averageSpeeds = np.min(vs, axis=1)

        self.__numpyMicrotypeSpeed[:, self.modeToIdx['auto']] = averageSpeeds
        # np.copyto(self.__numpySpeed[:, self.modeToIdx['auto']], averageSpeeds)
        indices = np.nonzero(self.__transitionMatrixNetworkIdx)[0]
        for idx, spd in zip(indices, averageSpeeds):
            self.__numpyNetworkSpeed[idx, self.modeToIdx['auto']] = spd
            np.copyto(self.__numpyInstantaneousSpeed[idx, :], vs[idx, :])
            np.copyto(self.__numpyInstantaneousAccumulation[idx, :], ns[idx, :])

        # for microtypeID, microtype in self:
        #     idx = self.transitionMatrix.idx(microtypeID)
        #     for modes, autoNetwork in microtype.networks:
        #         if "auto" in autoNetwork:
        #             networkStateData = collectedNetworkStateData[(microtypeID, modes)]
        #             networkStateData.finalAccumulation = ns[idx, -1]
        #             networkStateData.finalSpeed = vs[idx, -1]
        #             networkStateData.averageSpeed = averageSpeeds[idx]
        #             networkStateData.inflow = np.squeeze(inflows[idx, :]) * dt
        #             networkStateData.outflow = np.squeeze(outflows[idx, :]) * dt
        #             networkStateData.flowMatrix = np.squeeze(flowMats[idx, :, :]) * dt
        #             networkStateData.n = np.squeeze(ns[idx, :])
        #             networkStateData.v = np.squeeze(vs[idx, :])
        #             networkStateData.t = np.squeeze(ts) + networkStateData.initialTime
        return {"t": np.transpose(ts), "v": np.transpose(vs), "n": np.transpose(ns),
                "max_accumulation": 100}

    def __iter__(self) -> (str, Microtype):
        return iter(self.__microtypes.items())

    def getModeSpeeds(self) -> dict:
        return {mode: {microtypeId: self.__numpyMicrotypeSpeed[microtypeIdx, modeIdx] for microtypeId, microtypeIdx in
                       self.microtypeIdToIdx.items()} for mode, modeIdx in self.modeToIdx.items()}

    def getOperatorCosts(self) -> CollectedTotalOperatorCosts:
        operatorCosts = CollectedTotalOperatorCosts()
        for mID, microtype in self:
            operatorCosts[mID] = microtype.networks.getModeOperatingCosts()
        return operatorCosts

    def getStateData(self) -> CollectedNetworkStateData:
        data = CollectedNetworkStateData()
        for mID, microtype in self:
            data.addMicrotype(microtype)
        return data

    def importPreviousStateData(self, networkStateData: CollectedNetworkStateData):
        for mID, microtype in self:
            networkStateData.adoptPreviousMicrotypeState(microtype)

    def resetStateData(self):
        for _, nsd in self.collectedNetworkStateData:
            nsd.reset()

    # def updateTransitionMatrix(self, transitionMatrix: TransitionMatrix):
    #     if self.transitionMatrix.names == transitionMatrix.names:
    #         self.transitionMatrix = transitionMatrix
    #     else:
    #         print("MICROTYPE NAMES IN TRANSITION MATRIX DON'T MATCH")

    def emptyTransitionMatrix(self):
        return TransitionMatrix(self.transitionMatrix.names)

    def filterAllocation(self, mode, inputAllocation: Allocation):
        validMicrotypes = self.__modeToMicrotype[mode]
        if inputAllocation.keys() == validMicrotypes:
            return inputAllocation.mapping
        else:
            return inputAllocation.filterAllocation(validMicrotypes)


@njit(fastmath=True, parallel=False, cache=True)
def vectorV(N, n_other, L_eff, speedFunctions, minspeed=0.005):
    nTimeSteps = N.shape[1]
    out = np.empty_like(N)

    def v(n, n_other, L_eff, speedFunctions):
        density = (n + n_other) / L_eff
        v_out = np.zeros_like(n)
        for ind, d in enumerate(density):
            v_out[ind] = speedFunctions[ind](d)
        return v_out

    for t in np.arange(nTimeSteps):
        n = N[:, t]
        out[:, t] = v(n, n_other, L_eff, speedFunctions)

    return out


@njit(fastmath=True, parallel=False, cache=True)
def doMatrixCalcs(N, n_init, Xprime, tripStartRate, characteristicL, L_eff, n_other, dt, speedFunctions):
    X = np.transpose(Xprime)
    nTimeSteps = N.shape[1]

    N[:, 0] = n_init

    def v(n, n_other, L_eff, speedFunctions):
        density = (n + n_other) / L_eff
        v_out = np.zeros_like(n)
        for ind, d in enumerate(density):
            v_out[ind] = speedFunctions[ind](d)
        return v_out

    def outflow(n):
        return v(n, n_other, L_eff, speedFunctions) * n / characteristicL

    def inflow(n):
        return X @ (v(n, n_other, L_eff, speedFunctions) * n / characteristicL)

    def spillback(n, demand, inflow, outflow, dt):
        requestedN = (demand + inflow - outflow) * dt + n
        return requestedN

    for t in np.arange(nTimeSteps - 1):
        n_t = N[:, t]
        infl = inflow(n_t)
        outfl = outflow(n_t)
        n_t = spillback(n_t, tripStartRate, infl, outfl, dt)
        n_t[n_t < 0] = 0.0
        N[:, t + 1] = n_t

    return N
