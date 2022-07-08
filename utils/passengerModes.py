import numpy as np
import pandas as pd
from utils.supply import TravelDemand
from utils.data import ScenarioData

mph2mps = 1609.34 / 3600


class Mode:
    def __init__(self, networks, params, microtypeID, microtypePopulation, name, travelDemandData, microtypeSpeed,
                 microtypeCosts, fleetSize, accessDistance, diToIdx):
        self.name = name
        self.params = params
        self.microtypeID = microtypeID
        self.microtypePopulation = microtypePopulation
        self._idx = params.index.get_loc(microtypeID)
        self.modeParamsColumnToIdx = {i: params.columns.get_loc(i) for i in params.columns}
        self._params = params.to_numpy()
        self.networks = networks
        self.microtypeSpeed = microtypeSpeed
        self.microtypeCosts = microtypeCosts
        self.fleetSize = fleetSize
        self.accessDistance = accessDistance
        self.diToIdx = diToIdx
        self.__fareIdx = (0, 0)
        self.__discountFareIdx = (0, 0)
        self.defineCosts()

        self._networkSpeed = [n.getModeNetworkSpeed(name) for n in networks]
        self._networkOperatingSpeed = [n.getModeOperatingSpeed(name) for n in networks]
        self._networkBlockedDistance = [n.getModeBlockedDistance(name) for n in networks]
        self._networkAccumulation = [n.getModeAccumulation(name) for n in networks]
        self._networkLength = [n.networkLength for n in networks]

        self._averagePassengerDistanceInSystem = 0.0

        self.__bad = False
        self.fixedVMT = True

        self.travelDemand = TravelDemand(travelDemandData)
        self._PMT = self.travelDemand.rateOfPmtPerHour
        self._VMT = 0.

    def defineCosts(self):
        if "SeniorFareDiscount" in self.params.columns:
            seniorDIs = np.array([di.isSenior() for di in self.diToIdx.keys()])
            self.__fareIdx = (np.where(~seniorDIs)[0][0], 0)
            self.__discountFareIdx = (np.where(seniorDIs)[0][0], 0)
        else:
            self.__fareIdx = (0, 0)
            self.__discountFareIdx = (0, 0)
        if np.any(self.microtypeCosts):
            pass
        else:
            self.microtypeCosts[:, ScenarioData.costTypeToIdx["perStartPublicCost"]] = self.params.at[
                self.microtypeID, "PerStartCost"].copy()
            self.microtypeCosts[:, ScenarioData.costTypeToIdx["perEndPrivateCost"]] = self.params.at[
                self.microtypeID, "PerEndCost"].copy()
            self.microtypeCosts[:, ScenarioData.costTypeToIdx["perMilePrivateCost"]] = self.params.at[
                self.microtypeID, "PerMileCost"].copy()
            self.accessDistance.fill(self.getAccessDistance())
            if "SeniorFareDiscount" in self.params.columns:
                seniorDIs = np.array([di.isSenior() for di in self.diToIdx.keys()])
                self.microtypeCosts[seniorDIs, ScenarioData.costTypeToIdx["perStartPublicCost"]] *= self.params.at[
                    self.microtypeID, "SeniorFareDiscount"]
        if "Headway" in self.params.columns:
            self.microtypeCosts[:, ScenarioData.costTypeToIdx["waitTimeInSeconds"]] = self.params.at[
                                                                                          self.microtypeID, "Headway"].copy() / 2.0

    @property
    def fare(self):
        return self.microtypeCosts[self.__fareIdx]

    @property
    def discountFare(self):
        return self.microtypeCosts[self.__discountFareIdx]

    @property
    def relativeLength(self):
        # return self.params.to_numpy()[self._inds["VehicleSize"]]
        return self.params.at[self.microtypeID, "VehicleSize"]

    def updateScenarioInputs(self):
        pass

    def updateRouteAveragedSpeed(self):
        pass

    def getDemandForVmtPerHour(self):
        return self.travelDemand.rateOfPmtPerHour * self.relativeLength

    def getAccessDistance(self) -> float:
        return 0.0

    def updateModeBlockedDistance(self):
        pass

    def getSpeedDifference(self, allocation: list):
        speeds = np.array([n.NEF(a * self._VMT * mph2mps, self.name) for n, a in zip(self.networks, allocation)])
        return np.linalg.norm(speeds - np.mean(speeds))

    def assignVmtToNetworks(self):
        Ltot = sum(self._networkLength)[0]
        for ind, n in enumerate(self.networks):
            VMT = self._PMT * self._networkLength[ind] / Ltot
            self._networkAccumulation[ind] = VMT / self._networkOperatingSpeed[ind]

    def __str__(self):
        return str([self.name + ': VMT=' + str(self._PMT) + ', L_blocked=' + str(self._networkBlockedDistance)])

    def getSpeed(self):
        return self.networks[0].getBaseSpeed()

    def getNs(self):
        return self._networkAccumulation

    def getPassengerFlow(self) -> float:
        if np.any([n.isJammed for n in self.networks]):
            return 0.0
        else:
            return self.travelDemand.rateOfPmtPerHour

    def updateHeadway(self, newHeadway):
        pass

    def getOperatorCosts(self) -> float:
        return 0.0

    def getOperatorRevenues(self) -> float:
        return 0.0

    def getPortionDedicated(self) -> float:
        return 0.0


class WalkMode(Mode):
    def __init__(self, networks, modeParams: pd.DataFrame, microtypeID: str, microtypePopulation: float,
                 travelDemandData=None, speedData=None, microtypeCosts=None, fleetSize=None, accessDistance=None,
                 diToIdx=None) -> None:
        super(WalkMode, self).__init__(networks=networks, params=modeParams, microtypeID=microtypeID,
                                       microtypePopulation=microtypePopulation, name="walk",
                                       travelDemandData=travelDemandData, microtypeSpeed=speedData,
                                       microtypeCosts=microtypeCosts, fleetSize=fleetSize,
                                       accessDistance=accessDistance, diToIdx=diToIdx)
        self.fixedVMT = False
        for n in networks:
            n.addPassengerMode(self)
            n.setModeNetworkSpeed(self.name, self.speedInMetersPerSecond)
            n.setModeOperatingSpeed(self.name, self.speedInMetersPerSecond)

    @property
    def perStart(self):
        # return self.params.to_numpy()[self._inds["PerStartCost"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["PerEndCost"]]

    @property
    def perEnd(self):
        # return self.params.to_numpy()[self._inds["PerEndCost"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["PerEndCost"]]

    @property
    def perMile(self):
        # return self.params.to_numpy()[self._inds["PerMileCost"]]
        return self.params.at[self.microtypeID, "PerMileCost"]

    @property
    def speedInMetersPerSecond(self):
        # return self.params.to_numpy()[self._inds["PerEndCost"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["SpeedInMetersPerSecond"]]

    def getSpeed(self):
        return self.speedInMetersPerSecond


class BikeMode(Mode):
    def __init__(self, networks, modeParams: pd.DataFrame, microtypeID: str, microtypePopulation: float,
                 travelDemandData=None, speedData=None, microtypeCosts=None, fleetSize=None, accessDistance=None,
                 diToIdx=None) -> None:
        super(BikeMode, self).__init__(networks=networks, params=modeParams, microtypeID=microtypeID,
                                       microtypePopulation=microtypePopulation, name="bike",
                                       travelDemandData=travelDemandData, microtypeSpeed=speedData,
                                       microtypeCosts=microtypeCosts, fleetSize=fleetSize,
                                       accessDistance=accessDistance, diToIdx=diToIdx)
        self.fixedVMT = False
        for n in networks:
            n.addPassengerMode(self)
            n.setModeNetworkSpeed(self.name, self.speedInMetersPerSecond)
            n.setModeOperatingSpeed(self.name, self.speedInMetersPerSecond)
        self.bikeLanePreference = 2.0
        self.sharedFleetSize = self._params[
            self._idx, self.modeParamsColumnToIdx["BikesPerCapita"]]  # TODO: Rename to density

    @property
    def sharedFleetSize(self):
        return self.fleetSize[0]

    @sharedFleetSize.setter
    def sharedFleetSize(self, newSize):
        self.fleetSize.fill(newSize)

    @property
    def dailyOperatingCostPerBike(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["DailyOpCostPerBike"]]

    @property
    def perStart(self):
        # return self.params.to_numpy()[self._inds["PerStartCost"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["PerEndCost"]]

    @property
    def perEnd(self):
        # return self.params.to_numpy()[self._inds["PerEndCost"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["PerEndCost"]]

    @property
    def perMile(self):
        # return self.params.to_numpy()[self._inds["PerMileCost"]]
        return self.params.at[self.microtypeID, "PerMileCost"]

    @property
    def speedInMetersPerSecond(self):
        # return self.params.to_numpy()[self._inds["PerEndCost"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["SpeedInMetersPerSecond"]]

    @property
    def dedicatedLanePreference(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["DedicatedLanePreference"]]

    def getOperatorCosts(self) -> float:
        # TODO: Check what units sharedFleetSize is in
        return self.sharedFleetSize * self.microtypePopulation * self.dailyOperatingCostPerBike / 1000.0

    def getSpeed(self):
        return self.speedInMetersPerSecond

    def distanceOnDedicatedLanes(self, capacityTot, capacityDedicated) -> (float, float):
        capacityMixed = capacityTot - capacityDedicated
        portionMixed = (
                capacityDedicated / capacityTot + self.dedicatedLanePreference * capacityMixed / capacityTot)
        N = self._PMT[0] / self.speedInMetersPerSecond
        N_dedicated = min([capacityDedicated, (1 - portionMixed) * N])
        N_mixed = N - N_dedicated
        return N_dedicated * self.speedInMetersPerSecond, N_mixed * self.speedInMetersPerSecond

    def assignVmtToNetworks(self):
        capacityTot = sum([n.L * n.jamDensity for n in self.networks])
        capacityDedicated = sum([n.L * n.jamDensity for n in self.networks if n.dedicated])
        capacityMixed = capacityTot - capacityDedicated
        VMT_dedicated, VMT_mixed = self.distanceOnDedicatedLanes(capacityTot, capacityDedicated)
        for n in self.networks:
            if n.dedicated | (n.L == 0):
                if VMT_dedicated == 0:
                    VMT = 0
                else:
                    VMT = VMT_dedicated * n.L * n.jamDensity / capacityDedicated
            else:
                if VMT_mixed == 0:
                    VMT = 0
                else:
                    VMT = VMT_mixed * n.L * n.jamDensity / capacityMixed
            # self._VMT[n] = VMT
            # n.setVMT(self.name, self._VMT[n])
            # if np.isnan(VMT / n.getModeNetworkSpeed(self.name)):
            acc = VMT / n.getModeNetworkSpeed(self.name)
            n.setModeAccumulation(self.name, acc)
            # self._N_eff[n] = VMT / self._speed[n] * self.relativeLength
            # n.setN(self.name, self._N_eff[n])

    def getPortionDedicated(self) -> float:
        if self._PMT > 0:
            tot = 0.0
            tot_dedicated = 0.0
            for key, val in zip(self.networks, self._networkAccumulation):
                tot += val
                if key.dedicated:
                    tot_dedicated += val
            if tot == 0:
                return 1.0
            else:
                return tot_dedicated / tot
        else:
            return 0.0

    def updateScenarioInputs(self):
        pass
        # self._params = self.params.to_numpy()
        # for n in self.networks:
        # self._L_blocked[n] = 0.0
        # self._VMT[n] = 0.0
        # self._N_eff[n] = 0.0
        # self._speed[n] = n.base_speed
        # self.__operatingL[n] = self.updateOperatingL(n)


class RailMode(Mode):
    def __init__(self, networks, modeParams: pd.DataFrame, microtypeID: str, microtypePopulation: float,
                 travelDemandData=None, speedData=None, microtypeCosts=None, fleetSize=None, accessDistance=None,
                 diToIdx=None) -> None:
        super(RailMode, self).__init__(networks=networks, params=modeParams, microtypeID=microtypeID,
                                       microtypePopulation=microtypePopulation, name="rail",
                                       travelDemandData=travelDemandData, microtypeSpeed=speedData,
                                       microtypeCosts=microtypeCosts, fleetSize=fleetSize,
                                       accessDistance=accessDistance, diToIdx=diToIdx)
        self.fixedVMT = True
        for n in networks:
            n.addPassengerMode(self)
            n.setModeNetworkSpeed(self.name, n.base_speed)
            n.setModeOperatingSpeed(self.name, self.routeAveragedSpeed)

    @property
    def perStart(self):
        # return self.params.to_numpy()[self._inds["PerStartCost"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["PerEndCost"]]

    @property
    def perEnd(self):
        # return self.params.to_numpy()[self._inds["PerEndCost"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["PerEndCost"]]

    @property
    def perMile(self):
        # return self.params.to_numpy()[self._inds["PerMileCost"]]
        return self.params.at[self.microtypeID, "PerMileCost"]

    @property
    def routeAveragedSpeed(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["SpeedInMetersPerSecond"]]

    @property
    def vehicleOperatingCostPerHour(self):
        # return self.params.to_numpy()[self._inds["VehicleOperatingCostsPerHour"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["VehicleOperatingCostPerHour"]]

    @property
    def headwayInSec(self):
        # return self.params.to_numpy()[self._inds["Headway"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["Headway"]]

    @headwayInSec.setter
    def headwayInSec(self, value):
        self._params[self._idx, self.modeParamsColumnToIdx["Headway"]] = value

    @property
    def stopSpacingInMeters(self):
        # return self.params.to_numpy()[self._inds["StopSpacing"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["StopSpacing"]]

    @property
    def portionAreaCovered(self):
        return max([self._params[self._idx, self.modeParamsColumnToIdx["CoveragePortion"]], 0.01])
        # return self.params.at[self.microtypeID, "CoveragePortion"]

    @property
    def accessDistanceMultiplier(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["AccessDistanceMultiplier"]]

    @property
    def routeDistanceToNetworkDistance(self) -> float:
        return self._params[self._idx, self.modeParamsColumnToIdx["CoveragePortion"]]

    def updateDemand(self, travelDemand=None):
        if travelDemand is not None:
            self.travelDemand = travelDemand
        self._VMT_tot = self.getRouteLength() / self.headwayInSec

    def getAccessDistance(self) -> float:
        return self.stopSpacingInMeters * self.accessDistanceMultiplier / self.routeDistanceToNetworkDistance

    def getSpeed(self):
        return self.routeAveragedSpeed

    def assignVmtToNetworks(self):
        self._networkAccumulation[0] = self.getRouteLength() / self.getSpeed() / self.headwayInSec

    def getRouteLength(self):
        return sum([n.L for n in self.networks])

    def getOperatorCosts(self) -> float:
        return sum(self.getNs()) * self.vehicleOperatingCostPerHour

    def getOperatorRevenues(self) -> float:
        return self.travelDemand.tripStartRatePerHour * self.fare

    def getDemandForVmtPerHour(self):
        return self.getRouteLength() / self.headwayInSec * 3600.

    def updateScenarioInputs(self):
        self._params = self.params.to_numpy()

    def updateHeadway(self, newHeadway):
        self.headwayInSec = newHeadway
        self.microtypeCosts[:, ScenarioData.costTypeToIdx["waitTimeInSeconds"]] = newHeadway / 2.0


class AutoMode(Mode):
    def __init__(self, networks, modeParams: pd.DataFrame, microtypeID: str, microtypePopulation: float,
                 travelDemandData=None, speedData=None, microtypeCosts=None, fleetSize=None, accessDistance=None,
                 diToIdx=None) -> None:
        super(AutoMode, self).__init__(networks=networks, params=modeParams, microtypeID=microtypeID,
                                       microtypePopulation=microtypePopulation, name="auto",
                                       travelDemandData=travelDemandData, microtypeSpeed=speedData,
                                       microtypeCosts=microtypeCosts, fleetSize=fleetSize,
                                       accessDistance=accessDistance, diToIdx=diToIdx)
        self.MFDmode = "single"
        self.fixedVMT = False
        for n in networks:
            n.addPassengerMode(self)
            # n.setModeNetworkSpeed(self.name, n.base_speed)

    @property
    def perStart(self):
        # return self.params.to_numpy()[self._inds["PerStartCost"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["PerEndCost"]]

    @property
    def perEnd(self):
        # return self.params.to_numpy()[self._inds["PerEndCost"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["PerEndCost"]]

    @property
    def perMile(self):
        # return self.params.to_numpy()[self._inds["PerMileCost"]]
        return self.params.at[self.microtypeID, "PerMileCost"]

    @property
    def relativeLength(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["VehicleSize"]]

    def getSpeed(self):
        return self._speedData[0]

    def x0(self):
        return np.array([1. / len(self.networks)] * len(self.networks))

    def constraints(self):
        return dict(type='eq', fun=lambda x: sum(x) - 1.0, jac=lambda x: [1.0] * len(x))

    def bounds(self):
        return [(0.0, 1.0)] * len(self.networks)

    def updateDemand(self, travelDemand=None):  # TODO: Why did I add this?
        if travelDemand is None:
            travelDemand = self.travelDemand
        else:
            self.travelDemand = travelDemand
        self._VMT_tot = travelDemand.rateOfPmtPerHour * self.relativeLength

    # @profile
    def assignVmtToNetworks(self):
        # FIXME:  This doesn't work
        if len(self.networks) == 1:
            n = self.networks[0]
            n.setModeAccumulation(self.name, self._PMT / n.getModeNetworkSpeed(self.name))
            if self.MFDmode != "single":
                n.runSingleNetworkMFD()
        elif len(self.networks) > 1:
            raise NotImplementedError("Can't have multiple auto subnetworks yet")
            """         
            res = minimize(self.getSpeedDifference, self.x0(), constraints=self.constraints(), bounds=self.bounds())
            for ind, (n, a) in enumerate(zip(self.networks, res.x)):
                VMT = a * self._VMT_tot
                spd = n.NEF(a * self._VMT_tot * mph2mps, self.name)
                N_eff = VMT / spd
                self._VMT[n] = VMT
                self._speed[n] = spd
                n.setVMT(self.name, VMT)
                self._N_eff[n] = N_eff
                n.setN(self.name, self._N_eff[n])
                self._networkAccumulation[n][ind] = N_eff"""
        else:
            print("OH NO!")


class BusMode(Mode):
    def __init__(self, networks, modeParams: pd.DataFrame, microtypeID: str, microtypePopulation: float,
                 travelDemandData=None, speedData=None, microtypeCosts=None, fleetSize=None, accessDistance=None,
                 diToIdx=None) -> None:
        super().__init__(networks=networks, params=modeParams, microtypeID=microtypeID,
                         microtypePopulation=microtypePopulation, name="bus",
                         travelDemandData=travelDemandData, microtypeSpeed=speedData, microtypeCosts=microtypeCosts,
                         fleetSize=fleetSize, accessDistance=accessDistance, diToIdx=diToIdx)
        self.fixedVMT = True
        self.__operatingL = dict()
        self.__availableRoadNetworkDistance = sum([n.L for n in self.networks])

        for n in networks:
            n.addPassengerMode(self)
            self.__operatingL[n] = self.updateOperatingL(n)

        self.__routeLength = self.updateRouteLength()
        # self.travelDemand = TravelDemand(travelDemandData)
        self.routeAveragedSpeed = 10.0  # self.getSpeed()
        self.occupancy = 0.0
        self.updateModeBlockedDistance()
        self.__N = self.getN()

    @property
    def routeAveragedSpeed(self):
        return self.microtypeSpeed[0]

    @routeAveragedSpeed.setter
    def routeAveragedSpeed(self, spd):
        self.microtypeSpeed[0] = spd
        self.__N = self.getN()

    @property
    def desiredHeadwayInSec(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["Headway"]]

    @desiredHeadwayInSec.setter
    def desiredHeadwayInSec(self, headway):
        self._params[self._idx, self.modeParamsColumnToIdx["Headway"]] = headway

    @property
    def headwayInSec(self):
        return (self.__routeLength / self.routeAveragedSpeed) / self.__N

    @property
    def passengerWaitInSec(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["PassengerWait"]]

    @property
    def passengerWaitInSecDedicated(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["PassengerWaitDedicated"]]

    @property
    def stopSpacingInMeters(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["StopSpacing"]]

    @property
    def minStopTimeInSec(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["MinStopTime"]]

    @property
    def perStart(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["PerStartCost"]]

    @property
    def perEnd(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["PerEndCost"]]

    @property
    def perMile(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["PerMileCost"]]

    @property
    def vehicleOperatingCostPerHour(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["VehicleOperatingCostPerHour"]]

    @property
    def accessDistanceMultiplier(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["AccessDistanceMultiplier"]]

    @property
    def routeDistanceToNetworkDistance(self) -> float:
        """
        Changed January 2021: Removed need for car-only subnnetworks.
        Now buses only run on a fixed portion of the bus/car subnetwork
        """
        return self._params[self._idx, self.modeParamsColumnToIdx["CoveragePortion"]]

    @property
    def relativeLength(self):
        # return self.params.to_numpy()[self._inds["VehicleSize"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["VehicleSize"]]

    def updateHeadway(self, newHeadway):
        self.desiredHeadwayInSec = newHeadway
        self.microtypeCosts[:, ScenarioData.costTypeToIdx["waitTimeInSeconds"]] = newHeadway / 2.0

    def updateScenarioInputs(self):
        self._params = self.params.to_numpy()
        for n in self.networks:
            self.__operatingL[n] = self.updateOperatingL(n)
        self.__routeLength = self.updateRouteLength()

    def getAccessDistance(self) -> float:
        """Order of magnitude estimate for average walking distance to nearest stop"""
        return self.stopSpacingInMeters * self.accessDistanceMultiplier / self.routeDistanceToNetworkDistance

    def getDemandForVmtPerHour(self):
        return self.getRouteLength() / self.headwayInSec * 3600. / 1609.34

    def getOperatingL(self, network=None):
        if network is None:
            return self.__operatingL
        else:
            return self.__operatingL[network]

    def updateOperatingL(self, network) -> float:
        """Changed January 2021: Buses only operate on a portion of subnetwork"""
        if network.dedicated:
            return network.L
        else:
            dedicatedDistanceToBus = sum([n.L for n in self.networks if n.dedicated])
            totalDistance = self.__availableRoadNetworkDistance
            dedicatedDistanceToOther = totalDistance - sum([n.L for n in self.networks])
            undedicatedDistance = totalDistance - dedicatedDistanceToBus
            return max(0, (self.routeDistanceToNetworkDistance * totalDistance - dedicatedDistanceToBus) * (
                    network.L / (undedicatedDistance - dedicatedDistanceToOther)))
            # return self.routeDistanceToNetworkDistance * totalDistance * network.L / (totalDistance - dedicatedDistance)

    def getN(self, network=None):
        """Changed January 2021: Buses only operate on a portion of subnetwork"""
        if network:
            desiredN = self.getOperatingL(network) / self.routeAveragedSpeed / self.desiredHeadwayInSec
            maximumN = self.__operatingL[network] / network.avgLinkLength
        else:
            desiredN = self.getRouteLength() / self.routeAveragedSpeed / self.desiredHeadwayInSec
            maximumN = sum([self.__operatingL[n] / n.avgLinkLength for n in self.networks])
        if desiredN <= maximumN:
            return desiredN
        else:
            return maximumN

    def getRouteLength(self):
        return self.__routeLength

    def updateRouteLength(self):
        return sum([n.L for n in self.networks]) * self.routeDistanceToNetworkDistance

    def getSubNetworkSpeed(self, network):
        if network.dedicated:
            perPassenger = self.passengerWaitInSecDedicated
        else:
            perPassenger = self.passengerWaitInSec
        autoSpeed = network.autoSpeed
        numberOfStopsInSubnetwork = self.getOperatingL(network) / self.stopSpacingInMeters
        numberOfStopsInRoute = self.getRouteLength() / self.stopSpacingInMeters
        pass_per_stop = (self.travelDemand.tripStartRatePerHour + self.travelDemand.tripEndRatePerHour
                         ) / numberOfStopsInRoute * self.headwayInSec / 3600.
        stopping_time = numberOfStopsInSubnetwork * self.minStopTimeInSec
        stopped_time = perPassenger * pass_per_stop * numberOfStopsInSubnetwork + stopping_time
        driving_time = self.getOperatingL(network) / autoSpeed
        spd = self.getOperatingL(network) / (stopped_time + driving_time)
        if np.isnan(spd):
            spd = 0.001
            self.__bad = True
        else:
            self.__bad = False
        return spd

    def updateSubNetworkOperatingSpeeds(self):
        for n in self.networks:
            if n.L == 0:
                n.setModeOperatingSpeed(self.name, np.inf)
            else:
                bus_speed = self.getSubNetworkSpeed(n)
                n.setModeOperatingSpeed(self.name, bus_speed)

    def getSpeed(self):

        meters = np.zeros(len(self.networks), dtype=float)
        seconds = np.zeros(len(self.networks), dtype=float)
        speeds = np.zeros(len(self.networks), dtype=float)
        for idx, n in enumerate(self.networks):
            if n.L > 0:
                bus_speed = n.getModeOperatingSpeed(self.name)
                meters[idx] = self.getOperatingL(n)
                seconds[idx] = self.getOperatingL(n) / bus_speed
                speeds[idx] = meters[idx] / seconds[idx]
        if np.sum(seconds) > 0:
            spd = np.sum(meters) / np.sum(seconds)
            out = spd
        else:
            out = next(iter(self.networks)).getBaseSpeed()
        return out

    def updateRouteAveragedSpeed(self):
        self.routeAveragedSpeed = self.getSpeed()

    def calculateBlockedDistance(self, network) -> float:
        if network.dedicated:
            perPassenger = self.passengerWaitInSecDedicated
        else:
            perPassenger = self.passengerWaitInSec
        # bs = network.base_speed
        if network.autoSpeed > 0:
            numberOfStops = self.getRouteLength() / self.stopSpacingInMeters
            # numberOfBuses = self.getN(network)
            meanTimePerStop = (self.minStopTimeInSec + self.headwayInSec * perPassenger * (
                    self.travelDemand.tripStartRatePerHour + self.travelDemand.tripEndRatePerHour) / (
                                       numberOfStops * 3600.0))
            portionOfTimeStopped = min([meanTimePerStop * meanTimePerStop / self.headwayInSec, 1.0])
            # TODO: Think through this more fully. Is this the right way to scale up this time to distance?
            out = portionOfTimeStopped * network.avgLinkLength * self.getN(network)
            out = min(out, numberOfStops * network.avgLinkLength / 100,
                      (network.L - network.getBlockedDistance()) * 0.5)
            # portionOfRouteBlocked = out / self.routeLength
        else:
            out = 0
        return out

    def updateModeBlockedDistance(self):
        for n in self.networks:
            L_blocked = self.calculateBlockedDistance(n)
            n.setModeBlockedDistance(self.name, L_blocked)

    # @profile
    def assignVmtToNetworks(self):
        speeds = self._networkOperatingSpeed
        times = []
        lengths = []
        for ind, n in enumerate(self.networks):
            spd = speeds[ind]
            if spd < 0.1:
                # print("Speed to small: ", spd)
                spd = 0.1
            times.append(self.getOperatingL(n) / spd)
            lengths.append(self.getOperatingL(n))
        for ind, n in enumerate(self.networks):
            if speeds[ind] >= 0:
                self._networkOperatingSpeed[ind][0] = self.getSubNetworkSpeed(n)
                networkAccumulation = self.__N * times[ind] / sum(times)
                self._networkAccumulation[ind][0] = min(
                    networkAccumulation, self.getRouteLength() / n.avgLinkLength / 2 * self.relativeLength)
                if n.dedicated & (networkAccumulation > 0):
                    n.runSingleNetworkMFD()
        self.updateSubNetworkOperatingSpeeds()
        self.updateModeBlockedDistance()

    def getOccupancy(self) -> float:
        return self.travelDemand.averageDistanceInSystemInMiles / (
                self.routeAveragedSpeed * 2.23694) * self.travelDemand.tripStartRatePerHour / self.getN()

    def getPassengerFlow(self) -> float:
        if np.any([n.isJammed for n in self.networks]):
            return 0.0
        elif self.occupancy > 100:
            return np.nan
        else:
            return self.travelDemand.rateOfPmtPerHour

    def getOperatorCosts(self) -> float:
        return sum(self.getNs())[0] * self.vehicleOperatingCostPerHour

    def getOperatorRevenues(self) -> float:
        return (self.travelDemand.tripStartRatePerHour - self.travelDemand.discountTripStartRatePerHour
                ) * self.fare + self.travelDemand.discountTripStartRatePerHour * self.discountFare

    def getPortionDedicated(self) -> float:
        if self._PMT > 0:
            tot = 0.0
            tot_dedicated = 0.0
            for key, val in zip(self.networks, self._networkAccumulation):  # TODO: Take into account different speeds
                tot += val
                if key.dedicated:
                    tot_dedicated += val
            if tot == 0:
                return 0.
            else:
                return tot_dedicated / tot
        else:
            return 0.0
