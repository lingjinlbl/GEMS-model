#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy as sp
from numba import njit
from scipy.optimize import nnls

from .network import Network, NetworkCollection, Costs


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
            self.networks.getMode(mode).costs = modeCosts

    def updateNetworkSpeeds(self, nIters=None):
        self.networks.updateModes(nIters)

    def getModeSpeeds(self) -> dict:
        return {mode: self.getModeSpeed(mode) for mode in self.mode_names}

    def getModeSpeed(self, mode) -> float:
        return self.networks.getMode(mode).getSpeed()

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
        # self.networks.demands.resetDemand()

    def getModeStartAndEndRate(self, mode: str) -> (float, float):
        return self.networks.demands.getStartRate(mode), self.networks.demands.getStartRate(mode)

    def getModeStartRate(self, mode: str) -> float:
        return self.networks.demands.getStartRate(mode)

    def getModeMeanDistance(self, mode: str):
        return self.networks.demands.getAverageDistance(mode)

    # def addStartTimeCostWait(self, mode: str, cc: np.ndarray):  # Could eventually be vectorized
    #     if mode in self:
    #         perStartCost = self.networks.getMode(mode).perStart
    #         cc[self.paramToIdx['cost']] += perStartCost
    #         if mode in ['bus', 'rail']:
    #             waitTime = self.networks.getMode(mode).headwayInSec / 3600. / 4.  # TODO: Something better
    #             cc[self.paramToIdx['wait_time']] += waitTime
    #             accessTime = self.networks.getMode(
    #                 mode).getAccessDistance() / 1.5 / 3600.0  # TODO: Switch back to self.networks.modes['walk'].speedInMetersPerSecond
    #             cc[self.paramToIdx['access_time']] += accessTime
    #             if np.isnan(waitTime) | np.isnan(accessTime):
    #                 print('WHY IS THIS NAN')

    # def addEndTimeCostWait(self, mode: str, cc: np.ndarray):
    #     if mode in self:
    #         cc[self.paramToIdx['cost']] += self.networks.getMode(mode).perEnd
    #         if mode == 'bus':
    #             cc[self.paramToIdx['wait_time']] += self.networks.getMode('bus').headwayInSec / 3600. / 4.
    #             cc[self.paramToIdx['access_time']] += self.networks.getMode(mode).getAccessDistance() * \
    #                                                   self.networks.getMode(
    #                                                       'walk').speedInMetersPerSecond / 3600.0


def getFlows(self):
    return [mode.getPassengerFlow() for mode in self.networks.passengerModes.values()]


def getSpeeds(self):
    return [mode.getSpeed() for mode in self.networks.passengerModes.values()]


def getDemandsForPMT(self):
    return [mode.getPassengerFlow() for mode in
            self.networks.passengerModes.values()]


def __str__(self):
    return 'Demand: ' + str(self.getFlows()) + ' , Speed: ' + str(self.getSpeeds())


class MicrotypeCollection:
    def __init__(self, scenarioData, supplyData, demandData):
        self.__timeStepInSeconds = scenarioData.timeStepInSeconds
        self.__microtypes = dict()
        self.microtypePopulation = dict()
        self.__scenarioData = scenarioData
        self.modeData = scenarioData["modeData"]
        self.__firstFreightModeIdx = len(self.modeData)
        self.fleetData = scenarioData["fleetData"]
        # self.collectedNetworkStateData = CollectedNetworkStateData()
        self.__modeToMicrotype = dict()
        self.__networkIdToIdx = scenarioData.subNetworkIdToIdx
        self.__numpyDemand = supplyData['demandData']
        self.__numpyMicrotypeSpeed = supplyData['microtypeSpeed']
        self.__numpyFleetSize = supplyData[
            'fleetSize']  # Only nonzero when fleet size (rather than production) is fixed
        self.__numpyMicrotypeMixedTrafficDistance = supplyData['microtypeMixedTrafficDistance']
        self.__diameters = np.ndarray([0])
        self.__numpyNetworkAccumulation = supplyData['subNetworkAccumulation']
        self.__numpyNetworkLength = supplyData['subNetworkLength']
        # self.__numpyNetworkMaxAcceptance = supplyData['subNetworkMaxAcceptanceCapacity']
        self.__numpyScaledNetworkLength = supplyData['subNetworkScaledLength']
        self.__numpyNetworkMaxDensity = supplyData['subNetworkMaxDensity']
        self.__numpyNetworkVehicleSize = supplyData['subNetworkVehicleSize']
        self.__numpyNetworkSpeed = supplyData['subNetworkAverageSpeed']
        self.__numpyNetworkOperatingSpeed = supplyData['subNetworkOperatingSpeed']
        self.__numpyNetworkBlockedDistance = supplyData['subNetworkBlockedDistance']
        self.__numpyInstantaneousSpeed = supplyData['subNetworkInstantaneousSpeed']
        self.__numpyInstantaneousAccumulation = supplyData['subNetworkInstantaneousAutoAccumulation']
        self.__numpyInstantaneousQueueAccumulation = supplyData['subNetworkInstantaneousAutoQueueAccumulation']
        self.__accessDistance = supplyData['accessDistance']
        self.__microtypeCosts = supplyData['microtypeCosts']
        self.__numpyFreightProduction = supplyData['freightProduction']  # only set to nonzero when production is fixed
        self.__transitionMatrixNetworkIdx = supplyData['transitionMatrixNetworkIdx']
        self.__nonAutoModes = supplyData['nonAutoModes']
        self.__nInit = supplyData['subNetworkPreviousAutoAccumulation']
        self.__qInit = supplyData['subNetworkPreviousAutoQueueAccumulation']
        self.__subNetworkToMicrotype = supplyData['subNetworkToMicrotype']
        self.__MFDs = supplyData['MFDs']
        self.__maxInflow = supplyData['maxInflow']
        self.__tripRate = demandData['tripRate']
        self.__modeSplit = demandData['modeSplit']
        self.__toStarts = demandData['toStarts']
        self.__toDistanceByOrigin = demandData['toDistanceByOrigin']
        self.transitionMatrix = supplyData['transitionMatrix']

    def updateTransitionMatrix(self, newTransitionMatrix: np.ndarray):
        np.copyto(self.transitionMatrix, newTransitionMatrix)

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
    def passengerModeToIdx(self):
        return self.__scenarioData.passengerModeToIdx

    @property
    def freightModeToIdx(self):
        return self.__scenarioData.freightModeToIdx

    @property
    def demandDataTypeToIdx(self):
        return self.__scenarioData.demandDataTypeToIdx

    @property
    def paramToIdx(self):
        return self.__scenarioData.paramToIdx

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

    def updateNumpyPassengerDemand(self, data):
        self.__numpyDemand[:, :self.__firstFreightModeIdx, :] = data

    def updateNetworkData(self):
        for m in self.__microtypes.values():
            # assert isinstance(m, Microtype)
            # m.networks.updateNetworkData()
            m.networks.updateModeData()
            for n in m.networks:
                # assert isinstance(n, Network)
                n.updateScenarioInputs()

    def updateMFDparameter(self, subNetworkId, parameter, newValue):
        self.__scenarioData['subNetworkData'].loc[subNetworkId, parameter] = newValue
        for m in self.__microtypes.values():
            assert isinstance(m, Microtype)
            if subNetworkId in m.networks.subNetworkIDs():
                m.networks[subNetworkId].updateNetworkData()

    def recompileMFDs(self):
        for m in self.__microtypes.values():
            for n in m.networks:
                n.updateNetworkData()

    def __setitem__(self, key: str, value: Microtype):
        self.__microtypes[key] = value

    def __getitem__(self, item: str) -> Microtype:
        return self.__microtypes[item]

    def __contains__(self, item):
        return item in self.__microtypes

    def __len__(self):
        return len(self.__microtypes)

    def microtypeNames(self):
        return list(self.__microtypes.keys())

    def getModeStartRatePerSecond(self, mode):
        return np.array([microtype.getModeStartRate(mode) / 3600. for mID, microtype in self])

    def initializeFreightProduction(self):
        for (mID, freightMode), VMT in self.__scenarioData["freightDemand"].itertuples(index=True):
            self.__numpyFreightProduction[
                self.microtypeIdToIdx[mID], self.freightModeToIdx[freightMode]] = VMT  # TODO: Delete
            self.__numpyDemand[self.microtypeIdToIdx[mID], self.modeToIdx[freightMode], self.demandDataTypeToIdx[
                'vehicleDistance']] = VMT

    def importMicrotypes(self, override=False):
        # uniqueMicrotypes = subNetworkData["MicrotypeID"].unique()
        self.initializeFreightProduction()
        subNetworkData = self.__scenarioData["subNetworkData"]
        subNetworkCharacteristics = self.__scenarioData["subNetworkDataFull"]
        modeToSubNetworkData = self.__scenarioData["modeToSubNetworkData"]
        microtypeData = self.__scenarioData["microtypeIDs"]
        self.__diameters = np.zeros(len(microtypeData), dtype=float)
        for microtypeId, idx in self.microtypeIdToIdx.items():
            self.__diameters[idx] = microtypeData.loc[microtypeId, 'DiameterInMiles'] * 1609.34

        if len(self.__microtypes) == 0:
            self.__modeToMicrotype = dict()

        populationByMicrotype = self.__scenarioData["populations"].groupby("MicrotypeID").agg({"Population": sum})

        collectMatrixIds = (sum(self.__transitionMatrixNetworkIdx) == 0)
        for microtypeId, diameter in microtypeData.itertuples(index=False):
            self.microtypePopulation[microtypeId] = populationByMicrotype.loc[microtypeId].Population
            if (microtypeId in self) & ~override:
                self[microtypeId].resetDemand()
            else:
                subNetworkToModes = OrderedDict()
                modeToModeData = OrderedDict()
                allModes = set()
                for subNetworkId in subNetworkCharacteristics.loc[subNetworkCharacteristics[
                                                                      "MicrotypeID"] == microtypeId].index:
                    joined = modeToSubNetworkData.loc[
                        modeToSubNetworkData['SubnetworkID'] == subNetworkId]
                    self.__subNetworkToMicrotype[
                        self.microtypeIdToIdx[microtypeId], self.__networkIdToIdx[subNetworkId]] = True
                    subNetwork = Network(subNetworkData, subNetworkCharacteristics, subNetworkId, diameter, microtypeId,
                                         self.__numpyNetworkSpeed[self.__networkIdToIdx[subNetworkId], :],
                                         self.__numpyNetworkOperatingSpeed[self.__networkIdToIdx[subNetworkId], :],
                                         self.__numpyNetworkAccumulation[self.__networkIdToIdx[subNetworkId], :],
                                         self.__numpyNetworkBlockedDistance[self.__networkIdToIdx[subNetworkId], :],
                                         self.__numpyNetworkVehicleSize[self.__networkIdToIdx[subNetworkId], :],
                                         self.__numpyNetworkLength[self.__networkIdToIdx[subNetworkId], :],
                                         self.__numpyNetworkMaxDensity[self.__networkIdToIdx[subNetworkId], :],
                                         self.__MFDs[self.__networkIdToIdx[subNetworkId]],
                                         self.__maxInflow[self.__networkIdToIdx[subNetworkId]],
                                         self.modeToIdx)
                    self.__numpyScaledNetworkLength[self.__networkIdToIdx[subNetworkId], :] = self.__numpyNetworkLength[
                                                                                              self.__networkIdToIdx[
                                                                                                  subNetworkId], :]
                    if collectMatrixIds:
                        if 'auto' in subNetwork.modesAllowed.lower():  # Simple fix for now while we just have 1 auto network per microtype
                            self.__transitionMatrixNetworkIdx[self.__networkIdToIdx[subNetworkId]] = True

                    for n in joined.itertuples():
                        subNetworkToModes.setdefault(subNetwork, []).append(n.Mode.lower())
                        allModes.add(n.Mode.lower())
                        self.__modeToMicrotype.setdefault(n.Mode.lower(), set()).add(microtypeId)
                for mode in allModes:
                    if mode in self.modeData:
                        modeToModeData[mode] = self.modeData[mode]
                    else:
                        modeToModeData[mode] = self.fleetData[mode]
                netCol = NetworkCollection(subNetworkToModes, modeToModeData, microtypeId,
                                           self.microtypePopulation[microtypeId],
                                           self.__numpyDemand[self.microtypeIdToIdx[microtypeId], :, :],
                                           self.__numpyMicrotypeSpeed[self.microtypeIdToIdx[microtypeId], :],
                                           self.__microtypeCosts[self.microtypeIdToIdx[microtypeId], :, :, :],
                                           self.__numpyFleetSize[self.microtypeIdToIdx[microtypeId], :],
                                           self.__numpyFreightProduction[self.microtypeIdToIdx[microtypeId], :],
                                           self.__accessDistance[self.microtypeIdToIdx[microtypeId], :],
                                           self.demandDataTypeToIdx, self.modeToIdx, self.freightModeToIdx,
                                           self.diToIdx)
                self[microtypeId] = Microtype(microtypeId, netCol, self.paramToIdx)
                # self.collectedNetworkStateData.addMicrotype(self[microtypeId])

    def updateDedicatedDistance(self):
        for microtypeID, microtype in self:
            idx = self.microtypeIdToIdx[microtypeID]
            for mode in microtype.mode_names:
                portionDedicated = microtype.networks.getMode(mode).getPortionDedicated()
                distanceMixed = (1. - portionDedicated)  # microtype.getModeDemandForPMT(mode) *
                self.__numpyMicrotypeMixedTrafficDistance[idx, self.modeToIdx[mode]] = distanceMixed

    def transitionMatrixMFD(self, durationInHours):
        tripStartRatePerSecond = self.tripStartRateByMode[:, self.modeToIdx['auto']] / 3600

        # autoDistanceByODI = (self.__tripRate * self.__modeSplit[:, :, self.modeToIdx['auto']]).sum(
        #     axis=0) * self.__toDistanceByOrigin.sum(axis=1)

        # averageDistanceByStart = np.einsum("i,ij->j", autoDistanceByODI, self.__toStarts) / tripStartRatePerSecond

        characteristicL = np.zeros((len(self)), dtype=float)
        L_eff = np.zeros((len(self)), dtype=float)
        # n_init = np.zeros((len(self)), dtype=float)
        speedFunctions = [None] * len(self)
        inflowFunctions = [None] * len(self)
        for microtypeID, microtype in self:
            idx = self.microtypeIdToIdx[microtypeID]
            for autoNetwork in microtype.networks:
                if "auto" in autoNetwork:
                    characteristicL[idx] += autoNetwork.diameter * 1609.34
                    # n_init[idx] = networkStateData.initialAccumulation
                    speedFunctions[idx] = autoNetwork.MFD
                    inflowFunctions[idx] = autoNetwork.maxInflow
        n_init = self.__nInit[self.__transitionMatrixNetworkIdx]
        q_init = self.__qInit[self.__transitionMatrixNetworkIdx]
        L_blocked = self.__numpyNetworkBlockedDistance[self.__transitionMatrixNetworkIdx, :].sum(axis=1)
        L_eff = self.__numpyScaledNetworkLength[self.__transitionMatrixNetworkIdx, 0] - L_blocked
        jam_density = self.__numpyNetworkMaxDensity[self.__transitionMatrixNetworkIdx, 0]
        # MaxInflow = self.__numpyNetworkMaxAcceptance[self.__transitionMatrixNetworkIdx]
        n_other = (self.__numpyNetworkAccumulation[self.__transitionMatrixNetworkIdx, :][:,
                   self.nonAutoModes] * self.__numpyNetworkVehicleSize[self.__transitionMatrixNetworkIdx, :][:,
                                        self.nonAutoModes]).sum(axis=1)
        dt = self.__timeStepInSeconds

        # // TODO: other ways of having it sum to the correct value -- perhaps have it be proportional to the original characteristic L
        # expectedAverageTripDistance = np.linalg.inv(
        #     (np.eye(len(characteristicL)) - self.transitionMatrix).T) @ characteristicL

        # otherCharacteristicL = np.squeeze((np.eye(6) - self.transitionMatrix).T @ averageDistanceByStart[:, None])
        #
        # otherCharacteristicL = characteristicL * (otherCharacteristicL.sum() / characteristicL.sum())

        # otherCharacteristicL = characteristicL / expectedAverageTripDistance.sum() * averageDistanceByStart.sum()

        ts = np.arange(0, durationInHours * 3600., dt)
        ns = np.zeros((len(self), np.size(ts)), dtype=float)
        qs = np.zeros((len(self), np.size(ts)), dtype=float)
        # ns = doMatrixCalcs(ns, n_init, self.transitionMatrix, tripStartRatePerSecond, characteristicL,
        #                    L_eff,
        #                    n_other, dt, speedFunctions)
        # vs = vectorV(ns, n_other, L_eff, speedFunctions)

        ns, qs = doMatrixCalcs(ns, qs, n_init, q_init, self.transitionMatrix, tripStartRatePerSecond,
                               characteristicL, L_eff, n_other, jam_density, dt, speedFunctions, inflowFunctions)

        vs = vectorV(ns, qs, np.linalg.pinv(self.transitionMatrix.T), n_other, L_eff, jam_density, speedFunctions)

        averageSpeeds = np.sum(ns + qs, axis=1) / np.sum(ns / vs, axis=1)
        # production = (ns * vs).sum() * dt / 3600 * durationInHours
        # production_other = (ns_other * vs_other).sum() * dt / 3600 * durationInHours
        # production_requested = (averageDistanceByStart * tripStartRatePerSecond).sum() * durationInHours

        # averageSpeeds = np.min(vs, axis=1)

        self.__numpyMicrotypeSpeed[:, self.modeToIdx['auto']] = averageSpeeds

        indices = np.nonzero(self.__transitionMatrixNetworkIdx)[0]
        for i, (idx, spd) in enumerate(zip(indices, averageSpeeds)):
            self.__numpyNetworkSpeed[idx, :].fill(spd)
            self.__numpyNetworkOperatingSpeed[idx, self.modeToIdx['auto']] = spd
            np.copyto(self.__numpyInstantaneousSpeed[idx, :], vs[i, :])
            np.copyto(self.__numpyInstantaneousAccumulation[idx, :], ns[i, :])

        return {"t": np.transpose(ts), "v": np.transpose(vs), "n": np.transpose(ns),
                "max_accumulation": 100}

    def __iter__(self) -> (str, Microtype):
        return iter(self.__microtypes.items())

    def getModeSpeeds(self) -> dict:
        return {mode: {microtypeId: self.__numpyMicrotypeSpeed[microtypeIdx, modeIdx] for microtypeId, microtypeIdx in
                       self.microtypeIdToIdx.items()} for mode, modeIdx in self.modeToIdx.items()}

    def getOperatorCosts(self) -> np.ndarray:
        operatorCosts = np.zeros((len(self.microtypeIdToIdx), len(self.modeToIdx)))
        for mID, microtype in self:
            operatorCosts[self.microtypeIdToIdx[mID], :] = microtype.networks.getModeOperatingCosts()
        return operatorCosts

    def getFreightOperatorCosts(self) -> np.ndarray:
        operatorCosts = np.zeros((len(self.microtypeIdToIdx), len(self.modeToIdx)))
        for mID, microtype in self:
            operatorCosts[self.microtypeIdToIdx[mID], :] = microtype.networks.getFreightModeOperatingCosts()
        return operatorCosts


@njit(fastmath=True, parallel=False, cache=True)
def vectorV(N, Q, Xinv, n_other, L_eff, jam_density, speedFunctions, minspeed=0.005):
    nTimeSteps = N.shape[1]
    out = np.empty_like(N)

    def v(n, q, n_other, L_eff, jam_density, speedFunctions):
        queue_backup = Xinv @ q
        queue_backup[queue_backup < 0] = 0
        density = (n + n_other) / (L_eff - queue_backup * jam_density)
        v_out = np.zeros_like(n)
        for ind, d in enumerate(density):
            v_out[ind] = speedFunctions[ind](d)
        return v_out

    for t in np.arange(nTimeSteps):
        n = N[:, t]
        q = Q[:, t]
        out[:, t] = v(n, q, n_other, L_eff, jam_density, speedFunctions)

    return out


@njit(fastmath=True, parallel=False, cache=True)
def doMatrixCalcs(N, Q, n_init, q_init, Xprime, tripStartRate, characteristicL, L_eff, n_other, jam_density, dt,
                  speedFunctions, inflowFunctions):
    X = Xprime.T
    Xinv = np.linalg.pinv(X)
    nTimeSteps = N.shape[1]

    N[:, 0] = n_init
    Q[:, 0] = q_init

    def v(n, q, n_other, L_eff, jam_density, speedFunctions):
        queue_backup = Xinv @ q
        queue_backup[queue_backup < 0] = 0
        # queue_backup = sp.optimize.nnls(X, q)[0]
        density = (n + n_other) / (L_eff - queue_backup * jam_density)
        v_out = np.zeros_like(n)
        for ind, d in enumerate(density):
            v_out[ind] = speedFunctions[ind](d)
        return v_out

    def maxInflow(n, n_other, L_eff, inflowFunctions):
        density = (n + n_other) / L_eff
        flow_out = np.zeros_like(n)
        for ind, d in enumerate(density):
            flow_out[ind] = inflowFunctions[ind](d)
        return flow_out * L_eff / 3600.0

    def processOutflow(n, q):
        dn = -v(n, q, n_other, L_eff, jam_density, speedFunctions) * n / characteristicL * dt
        dn[dn > n] = n[dn > n]
        return dn

    def fillQueue(n, q, dn_outflow, tripStartRate):
        # 1: People start trips in the current location
        dn_start = tripStartRate * dt
        # 2: See how many people want to transfer into each subregion
        dn_inflow = (-X @ dn_outflow)
        # 3: calculate max inflow
        inflow_max = maxInflow(n, n_other, L_eff, inflowFunctions) * dt
        # 4: calculate inflow that must be turned back
        inflow_desired = dn_start + dn_inflow + q + dn_outflow
        inflow_actual = inflow_desired.copy()
        # if np.sum(inflow_actual > inflow_max) > 0:
        #     print('Turning back {0} vehicles'.format(
        #         int(np.sum(inflow_actual[inflow_actual > inflow_max] - inflow_max[inflow_actual > inflow_max]))))
        inflow_actual[inflow_actual > inflow_max] = inflow_max[inflow_actual > inflow_max]
        q_out = inflow_desired - inflow_actual
        return n + inflow_actual, q_out

    def drainQueue(n, q):

        dn = q
        dn[dn > mi] = mi[dn > mi]
        n = n + dn
        q = q - dn
        return n, q

    for t in np.arange(nTimeSteps - 1):
        n_t = N[:, t]
        q_t = Q[:, t]
        dn_t = processOutflow(n_t, q_t)
        n_t, q_t = fillQueue(n_t, q_t, dn_t, tripStartRate)
        # n_t, q_t = drainQueue(n_t, q_t)
        n_t[n_t < 0] = 0.0
        N[:, t + 1] = n_t
        Q[:, t + 1] = q_t

    # if np.any(Q > 0):
    #     print("Max")
    #     # print(Q.max(axis=1))
    #     print("End")
    #     print(Q[:, -1])
    #     # print("Queue stats: \n   Max: {0}\n   End: {1}".format(Q.max(axis=1), Q[:, -1]))

    return N, Q
