from typing import Dict

import numpy as np
import pandas as pd

from utils.OD import DemandIndex
from utils.choiceCharacteristics import ModalChoiceCharacteristics


class PopulationGroup:
    def __init__(self, homeLocation: str, populationGroupType: str, population: float):
        self.homeLocation = homeLocation
        self.populationGroupType = populationGroupType
        self.population = population


class DemandClass:
    def __init__(self, params: pd.DataFrame):
        self.__params = params.to_dict(orient="index")

    def __getitem__(self, item) -> float:
        item1, item2 = item
        if item1 in self.__params:
            return self.__params[item1].setdefault(item2, 0.0)
        else:
            return 0.0

    def updateModeSplit(self, mcc: ModalChoiceCharacteristics) -> Dict[str, float]:
        k = 1.0
        modes = mcc.modes()
        utils = np.zeros(len(modes), dtype=float)
        for idx, mode in enumerate(modes):
            util = 0.
            util += self[mode, "Intercept"]
            util += (mcc[mode].travel_time * 60.0) * self[mode, "BetaTravelTime"]
            util += (mcc[mode].wait_time * 60.0) * self[mode, "BetaWaitTime"]
            util += (mcc[mode].wait_time * 60.0) ** 2.0 * self[mode, "BetaWaitTimeSquared"]
            util += (mcc[mode].access_time * 60.0) * self[mode, "BetaAccessTime"]
            util += mcc[mode].cost * self[mode, "VOM"]
            if mode == "bike":
                util -= (mcc[mode].travel_time * 60.0) * self[mode, "BetaTravelTime"] * self[
                    mode, "ProtectedPreference"] * (mcc[mode].protected_distance / mcc.distanceInMiles)
            utils[idx] = util
            # utils = np.append(utils, util)
        exp_utils = np.exp(utils * k)
        probabilities = exp_utils / np.sum(exp_utils)
        mode_split = dict()
        for ind in range(np.size(probabilities)):
            mode_split[modes[ind]] = probabilities[ind]
        return mode_split

    def getModeCostPerTrip(self, mcc: ModalChoiceCharacteristics, mode, params=None):
        if mode not in mcc:
            return np.nan, np.nan, np.nan, np.nan
        if params is not None:
            params = DemandClass(params)
        else:
            params = self
        costPerTrip = 0.0
        inVehicleTime = 0.0
        outVehicleTime = 0.0
        distance = 0.0
        costPerTrip += params[mode, "Intercept"]
        costPerTrip += (mcc[mode].travel_time * 60.0) * params[mode, "BetaTravelTime"]
        costPerTrip += (mcc[mode].wait_time * 60.0) * params[mode, "BetaWaitTime"]
        costPerTrip += (mcc[mode].wait_time * 60.0) ** 2.0 * params[mode, "BetaWaitTimeSquared"]
        costPerTrip += (mcc[mode].access_time * 60.0) * self[mode, "BetaAccessTime"]
        costPerTrip += mcc[mode].cost * params[mode, "VOM"]
        inVehicleTime += mcc[mode].travel_time * 60.0
        outVehicleTime += mcc[mode].wait_time * 60.0 + mcc[mode].access_time * 60.0
        distance += mcc[mode].distance
        return costPerTrip, inVehicleTime, outVehicleTime, distance

    def getCostPerCapita(self, mcc: ModalChoiceCharacteristics, modeSplit, modes=None, params=None) -> (float, float):
        if modes is None:
            modes = modeSplit.keys()
        costPerCapita = 0.0
        totalDemandForTrips = 0.0
        inVehicleTime = 0.0
        outVehicleTime = 0.0
        distance = 0.0
        for mode in modes:
            split = modeSplit[mode]
            costPerTrip, inVehicle, outVehicle, dist = self.getModeCostPerTrip(mcc, mode, params)
            costPerCapita += costPerTrip * split
            inVehicleTime += inVehicle * split
            outVehicleTime += outVehicle * split
            distance += dist * split
            totalDemandForTrips += modeSplit.demandForTripsPerHour * split
        return costPerCapita, inVehicleTime, outVehicleTime, totalDemandForTrips, distance


class Population:
    """
    Class for storing and representing population of microtypes.
    """

    def __init__(self):
        self.__populationGroups = dict()
        self.__demandClasses = dict()
        self.__totalCosts = dict()
        self.totalPopulation = 0

    def __setitem__(self, key: DemandIndex, value: DemandClass):
        self.__demandClasses[key] = value

    def __getitem__(self, item: DemandIndex) -> DemandClass:
        return self.__demandClasses[item]

    def getPopulation(self, homeMicrotypeID: str, populationGroupType: str):
        if (homeMicrotypeID, populationGroupType) in self.__populationGroups:
            return self.__populationGroups[homeMicrotypeID, populationGroupType].population
        else:
            print("OH NO, no population group ", populationGroupType, " in microtype ", homeMicrotypeID)
            return 0

    def importPopulation(self, populations: pd.DataFrame, populationGroups: pd.DataFrame):
        for row in populations.itertuples():
            homeMicrotypeID = row.MicrotypeID
            populationGroupType = row.PopulationGroupTypeID
            self.__populationGroups[homeMicrotypeID, populationGroupType] = PopulationGroup(homeMicrotypeID,
                                                                                            populationGroupType,
                                                                                            row.Population)
            self.totalPopulation += row.Population
        for homeMicrotypeID in populations["MicrotypeID"].unique():
            for (groupId, tripPurpose), group in populationGroups.groupby(['PopulationGroupTypeID', 'TripPurposeID']):
                demandIndex = DemandIndex(homeMicrotypeID, groupId, tripPurpose)
                out = DemandClass(group.set_index("Mode").drop(columns=['PopulationGroupTypeID', 'TripPurposeID']))
                self[demandIndex] = out
        print("|  Loaded ", len(populations), " population groups")

    def __iter__(self):
        return iter(self.__demandClasses.items())
