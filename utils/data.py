import numpy as np
import os
import pandas as pd
from copy import deepcopy

from utils.OD import DemandIndex, ODindex


class ScenarioData:
    """
    Class to fetch and store data in a dictionary for specified scenario.

    ...

    Attributes
    ----------
    path : str
        File path to input data
    data : dict
        Dictionary containing input data from respective inputs

    Methods
    -------
    loadMoreData():
        Loads more modes of transportation.
    loadData():
        Read in data corresponding to various inputs.
    copy():
        Return a new ScenarioData copy containing data.
    """

    def __init__(self, path: str, timeStepInSeconds, data=None):
        """
        Constructs and loads all relevant data of the scenario into the instance.

        Parameters
        ----------
            path : str
                File path to input data
            data : dict
                Dictionary containing input data from respective inputs
        """
        self.__path = path
        self.__diToIdx = dict()  # TODO: Just define this once at the beginning of everything
        self.__odiToIdx = dict()
        self.__modeToIdx = dict()
        self.__passengerModeToIdx = dict()
        self.__freightModeToIdx = dict()
        self.__tripPurposeToIdx = dict()
        self.__demandDataTypeToIdx = dict()
        self.__microtypeIdToIdx = dict()
        self.__paramToIdx = dict()
        self.__transitLayerToIdx = dict()
        self.__subNetworkIdToIdx = dict()
        self.__distanceBinToDistance = dict()
        self.__populationGroupToIdx = dict()
        self.timeStepInSeconds = timeStepInSeconds
        if data is None:
            self.data = dict()
            self.loadData()
        else:
            self.data = data
            self.loadData()

    @property
    def paramToIdx(self):
        return self.__paramToIdx

    @property
    def diToIdx(self):
        return self.__diToIdx

    @property
    def odiToIdx(self):
        return self.__odiToIdx

    @property
    def modeToIdx(self):
        return self.__modeToIdx

    @property
    def demandDataTypeToIdx(self):
        return self.__demandDataTypeToIdx

    @property
    def passengerModeToIdx(self):
        return self.__passengerModeToIdx

    @property
    def freightModeToIdx(self):
        return self.__freightModeToIdx

    @property
    def microtypeIdToIdx(self):
        return self.__microtypeIdToIdx

    @property
    def subNetworkIdToIdx(self):
        return self.__subNetworkIdToIdx

    @property
    def distanceBinToDistance(self):
        return self.__distanceBinToDistance

    @property
    def microtypeIds(self):
        return list(self.__microtypeIdToIdx.keys())

    @property
    def transitLayerToIdx(self):
        return self.__transitLayerToIdx

    @property
    def firstFreightIdx(self):
        return len(self.passengerModeToIdx)

    @property
    def tripPurposeToIdx(self):
        return self.__tripPurposeToIdx

    @property
    def populationGroupToIdx(self):
        return self.__populationGroupToIdx

    def __setitem__(self, key: str, value):
        self.data[key] = value

    def __getitem__(self, item: str):
        return self.data[item]

    def loadModeData(self):
        """
        Follows filepath to modes and microtpe ID listed in the data files section under __path/modes.

        Returns
        -------
        A dict() with mode type as key and dataframe as value.
        """
        collected = dict()
        (_, _, fileNames) = next(os.walk(os.path.join(self.__path, "modes")))
        for file in fileNames:
            modeData = pd.read_csv(os.path.join(self.__path, "modes", file),
                                   dtype={"MicrotypeID": str}).set_index("MicrotypeID")
            if 'AccessDistanceMultiplier' not in modeData.columns:
                modeData['AccessDistanceMultiplier'] = 0.3
            collected[file.split(".")[0]] = modeData
        return collected

    def loadFleetData(self):
        collected = dict()
        (_, _, fileNames) = next(os.walk(os.path.join(self.__path, "fleets")))
        for file in fileNames:
            fleetData = pd.read_csv(os.path.join(self.__path, "fleets", file)).set_index("Fleet")
            for fleetName, data in fleetData.iterrows():
                collected[fleetName] = data
        return collected

    def loadData(self):
        """
        Fills the data dict() with values, the dict() contains data pertaining to various data labels and given csv
        data.
        """
        self["microtypeIDs"] = pd.read_csv(os.path.join(self.__path, "Microtypes.csv"),
                                           dtype={"MicrotypeID": str}).set_index("MicrotypeID", drop=False)
        subnetworkColumns = ["SubnetworkID", "Length", "vMax", "densityMax", "avgLinkLength",
                             "capacityFlow", "smoothingFactor", "waveSpeed", "a", "b",
                             "criticalDensity", "k_jam", "MicrotypeID"]
        renamedColumns = {"criticalDensity": "b", "densityMax": "k_jam"}
        subNetworkData = pd.read_csv(os.path.join(self.__path, "SubNetworks.csv"),
                                     usecols=lambda x: x in subnetworkColumns,
                                     index_col="SubnetworkID",
                                     dtype={"MicrotypeID": str}).fillna(0.0).rename(columns=renamedColumns)
        subNetworkData = subNetworkData.loc[subNetworkData.MicrotypeID.isin(self['microtypeIDs'].MicrotypeID)]
        if 'k_jam' in subNetworkData.columns:
            subNetworkData.loc[subNetworkData.k_jam == 0.0, 'k_jam'] = 0.15
        else:
            subNetworkData.k_jam = 0.15
        self["subNetworkData"] = subNetworkData.loc[:, ~subNetworkData.columns.duplicated()]
        subNetworkDataFull = pd.read_csv(os.path.join(self.__path, "SubNetworks.csv"),
                                         index_col="SubnetworkID", dtype={"MicrotypeID": str})
        self["subNetworkDataFull"] = subNetworkDataFull.loc[
            subNetworkDataFull.MicrotypeID.isin(self['microtypeIDs'].MicrotypeID)]
        self["modeToSubNetworkData"] = pd.read_csv(os.path.join(self.__path, "ModeToSubNetwork.csv"))
        self["microtypeAssignment"] = pd.read_csv(os.path.join(self.__path, "MicrotypeAssignment.csv"),
                                                  dtype={"FromMicrotypeID": str, "ToMicrotypeID": str,
                                                         "ThroughMicrotypeID": str}).fillna("None")
        self["populations"] = pd.read_csv(os.path.join(self.__path, "Population.csv"), dtype={"MicrotypeID": str})
        self["populationGroups"] = self.appendTripPurposeToPopulationGroups("PopulationGroups.csv", "TripPurposes.csv")
        self["timePeriods"] = pd.read_csv(os.path.join(self.__path, "TimePeriods.csv"))
        self["distanceBins"] = pd.read_csv(os.path.join(self.__path, "DistanceBins.csv"))
        self["originDestinations"] = pd.read_csv(os.path.join(self.__path, "OriginDestination.csv"),
                                                 dtype={"HomeMicrotypeID": str, "OriginMicrotypeID": str,
                                                        "DestinationMicrotypeID": str})
        self["distanceDistribution"] = pd.read_csv(os.path.join(self.__path, "DistanceDistribution.csv"),
                                                   dtype={"OriginMicrotypeID": str, "DestinationMicrotypeID": str})
        self["tripGeneration"] = pd.read_csv(os.path.join(self.__path, "TripGeneration.csv"))
        self["transitionMatrices"] = pd.read_csv(os.path.join(self.__path, "TransitionMatrices.csv"),
                                                 dtype={"OriginMicrotypeID": str, "DestinationMicrotypeID": str,
                                                        "From": str}).set_index(
            ["OriginMicrotypeID", "DestinationMicrotypeID", "DistanceBinID", "From"])
        self["laneDedicationCost"] = pd.read_csv(os.path.join(self.__path, "LaneDedicationCost.csv"),
                                                 dtype={"MicrotypeID": str}).set_index(["MicrotypeID", "Mode"])
        self["modeData"] = self.loadModeData()
        self["fleetData"] = self.loadFleetData()

        self["modeExternalities"] = pd.read_csv(os.path.join(self.__path, "ModeExternalities.csv"),
                                                dtype={"MicrotypeID": str}).set_index(["MicrotypeID", "Mode"])
        self["modeAvailability"] = pd.read_csv(os.path.join(self.__path, "ModeAvailability.csv"),
                                               dtype={"OriginMicrotypeID": str, "DestinationMicrotypeID": str})
        self["freightDemand"] = pd.read_csv(os.path.join(self.__path, "FreightDemand.csv"),
                                            dtype={"MicrotypeID": str}).set_index(["MicrotypeID", "Mode"])
        self["activityDensity"] = pd.read_csv(os.path.join(self.__path, "ActivityDensity.csv"),
                                              dtype={"MicrotypeID": str})
        self["tripPurposes"] = pd.read_csv(os.path.join(self.__path, "TripPurposes.csv")).set_index("TripPurposeID")
        self.defineIndices()

    def appendTripPurposeToPopulationGroups(self, populationGroups, tripPurposes):
        popGroupDf = pd.read_csv(os.path.join(self.__path, populationGroups))
        tripPurposeDf = pd.read_csv(os.path.join(self.__path, tripPurposes))
        if 'TripPurposeID' in popGroupDf.columns:
            # Add feature to check that they line up?
            return popGroupDf
        else:
            out = []
            for purp in tripPurposeDf.TripPurposeID:
                sub = popGroupDf.copy()
                sub['TripPurposeID'] = purp
                out.append(sub)
            return pd.concat(out)

    def defineIndices(self):
        self.__passengerModeToIdx = {mode: idx for idx, mode in enumerate(self["modeData"].keys())}
        self.__freightModeToIdx = {mode: idx for idx, mode in
                                   enumerate(self["freightDemand"].index.levels[1].to_list())}
        freightModeToGeneralIdx = {mode: idx + len(self.__passengerModeToIdx) for idx, mode in
                                   enumerate(self["freightDemand"].index.levels[1].to_list())}
        self.__modeToIdx = {**self.__passengerModeToIdx, **freightModeToGeneralIdx}
        # Starting in Python 3.9 we can re place with | operator

        self.__tripPurposeToIdx = {purpose: idx for idx, purpose in enumerate(self["tripPurposes"].index)}

        odJoinedToDistance = self['originDestinations'].merge(self['distanceDistribution'],
                                                              on=["OriginMicrotypeID", "DestinationMicrotypeID"],
                                                              suffixes=("_OD", "_Dist"), how="right")
        odJoinedToDistance = odJoinedToDistance.loc[odJoinedToDistance.OriginMicrotypeID.isin(
            self['microtypeIDs'].MicrotypeID) & odJoinedToDistance.DestinationMicrotypeID.isin(
            self['microtypeIDs'].MicrotypeID) & odJoinedToDistance.HomeMicrotypeID.isin(
            self['microtypeIDs'].MicrotypeID)]
        popGroupJoinedToTripGeneration = self.data['populations'].merge(self.data['tripGeneration'],
                                                                        on=['PopulationGroupTypeID'],
                                                                        suffixes=("_pop", "_trip"), how="inner")
        nestedDIs = popGroupJoinedToTripGeneration.groupby(
            ['MicrotypeID', 'PopulationGroupTypeID', 'TripPurposeID']).groups
        # nestedDIs = list(product(homeMicrotypeIDs, groupAndPurpose))
        DIs = [(hID, popGroup, purpose) for hID, popGroup, purpose in nestedDIs if
               hID in self['microtypeIDs'].MicrotypeID]
        self.__diToIdx = {DemandIndex(*di): idx for idx, di in enumerate(DIs)}
        self.__odiToIdx = {ODindex(*odi): idx for idx, odi in enumerate(
            odJoinedToDistance.groupby(['OriginMicrotypeID', 'DestinationMicrotypeID', 'DistanceBinID']).groups)}

        self.__demandDataTypeToIdx = {'tripStarts': 0, 'tripEnds': 1, 'passengerDistance': 3, 'vehicleDistance': 4,
                                      'discountTripStarts': 5}

        self.__microtypeIdToIdx = {mID: idx for idx, mID in enumerate(self["microtypeIDs"].MicrotypeID)}

        self.__subNetworkIdToIdx = {sID: idx for idx, sID in enumerate(self["subNetworkData"].index)}

        self.__distanceBinToDistance = {bd.DistanceBinID: bd.MeanDistanceInMiles for db, bd in
                                        self["distanceBins"].iterrows()}

        self.__paramToIdx = {'intercept': 0, 'travel_time': 1, 'cost': 2, 'wait_time': 3, 'access_time': 4,
                             'unprotected_travel_time': 5, 'distance': 6, 'mode_density': 7}
        uniqueTransitLayers = self.data['modeAvailability'].TransitLayer.unique()
        self.__transitLayerToIdx = {transitLayer: idx for idx, transitLayer in enumerate(uniqueTransitLayers)}
        self.__populationGroupToIdx = {grp: idx for idx, grp in
                                       enumerate(self['populationGroups']['PopulationGroupTypeID'].unique())}

    def copy(self):
        """
        Creates a deep copy of the data contained in this ScenarioData instance

        Returns
        -------
        A complete copy of the self.data dict()
        """
        return ScenarioData(self.__path, deepcopy(self.data))

    # def reallocate(self, fromSubNetwork, toSubNetwork, dist):

    def getPassengerModes(self):
        return set(self.__passengerModeToIdx.keys())


class ShapeParams:
    def __init__(self, scenarioData: ScenarioData, nSubBins, timeStepInSeconds):
        self.nSubBins = nSubBins
        self.timeStepInSeconds = timeStepInSeconds
        self.nTimePeriods = len(scenarioData['timePeriods']) * nSubBins
        self.nTimeSteps = int(scenarioData['timePeriods']['DurationInHours'].sum() * 3600 / timeStepInSeconds)
        self.nPassengerModes = len(scenarioData.passengerModeToIdx)
        self.nFreightModes = len(scenarioData.freightModeToIdx)
        self.nModesTotal = self.nPassengerModes + self.nFreightModes
        self.nDIs = len(scenarioData.diToIdx)
        self.nODIs = len(scenarioData.odiToIdx)
        self.nMicrotypes = len(scenarioData.microtypeIdToIdx)
        self.nSubNetworks = len(scenarioData['subNetworkData'])
        self.nParams = len(scenarioData.paramToIdx)
        self.nDemandDataTypes = len(scenarioData.demandDataTypeToIdx)
        self.nTransitLayers = len(scenarioData.transitLayerToIdx)
        self.nTripPurposes = len(scenarioData.tripPurposeToIdx)
        self.nPopulationGroups = len(scenarioData.populationGroupToIdx)


class Data:
    def __init__(self, scenarioData: ScenarioData, nSubBins, timeStepInSeconds):
        self.params = ShapeParams(scenarioData, nSubBins, timeStepInSeconds)
        self.scenarioData = scenarioData

        #############
        # Demand side
        #############

        self.__demandData = np.zeros(
            (self.params.nTimePeriods, self.params.nMicrotypes, self.params.nModesTotal,
             self.params.nDemandDataTypes))
        self.__modeSplit = np.zeros(
            (self.params.nTimePeriods, self.params.nDIs, self.params.nODIs, self.params.nPassengerModes), dtype=float)
        self.__tripRate = np.zeros(
            (self.params.nTimePeriods, self.params.nDIs, self.params.nODIs), dtype=float)
        self.__toStarts = np.zeros((self.params.nODIs, self.params.nMicrotypes), dtype=bool)
        self.__toEnds = np.zeros((self.params.nODIs, self.params.nMicrotypes), dtype=bool)
        self.__toThroughDistance = np.zeros((self.params.nODIs, self.params.nMicrotypes), dtype=float)
        self.__toDistanceByOrigin = np.zeros((self.params.nODIs, self.params.nMicrotypes), dtype=float)
        self.__toTransitLayer = np.zeros((self.params.nODIs, self.params.nTransitLayers), dtype=float)
        self.__utilities = np.zeros(
            (self.params.nTimePeriods, self.params.nDIs, self.params.nODIs, self.params.nPassengerModes), dtype=float)
        self.__choiceCharacteristics = np.zeros(
            (self.params.nTimePeriods, self.params.nDIs, self.params.nODIs, self.params.nPassengerModes,
             self.params.nParams),
            dtype=float)
        self.__toTripPurpose = np.zeros((self.params.nDIs, self.params.nTripPurposes), dtype=bool)
        self.__toHomeMicrotype = np.zeros((self.params.nDIs, self.params.nMicrotypes), dtype=bool)
        self.__toODI = np.zeros(
            (self.params.nDIs, self.params.nMicrotypes, self.params.nTripPurposes, self.params.nPopulationGroups),
            dtype=bool)
        self.__microtypeCosts = np.zeros(
            (self.params.nMicrotypes, self.params.nDIs, self.params.nModesTotal, 3), dtype=float)
        self.__transitLayerUtility = np.zeros((self.params.nPassengerModes, self.params.nTransitLayers), dtype=float)
        self.__choiceParameters = np.zeros((self.params.nDIs, self.params.nPassengerModes, self.params.nParams),
                                           dtype=float)
        self.__choiceParametersFixed = np.zeros((self.params.nDIs, self.params.nPassengerModes, self.params.nParams),
                                                dtype=float)
        self.__activityDensity = np.zeros((self.params.nMicrotypes, self.params.nTripPurposes), dtype=float)

        #############
        # Supply side
        #############
        self.__microtypeLengthMultiplier = np.ones((self.params.nTimePeriods, self.params.nMicrotypes), dtype=float)
        self.__subNetworkToMicrotype = np.zeros((self.params.nMicrotypes, self.params.nSubNetworks), dtype=bool)
        self.__microtypeSpeed = np.zeros(
            (self.params.nTimePeriods, self.params.nMicrotypes, self.params.nModesTotal))
        self.__transitionMatrix = np.zeros(
            (self.params.nTimePeriods, self.params.nMicrotypes, self.params.nMicrotypes))
        self.__transitionMatrices = np.zeros(
            (self.params.nODIs, self.params.nMicrotypes, self.params.nMicrotypes))
        self.__accessDistance = np.zeros((self.params.nMicrotypes, self.params.nModesTotal))
        self.__microtypeMixedTrafficDistance = np.zeros(
            (self.params.nTimePeriods, self.params.nMicrotypes, self.params.nModesTotal))
        self.__fleetSize = np.zeros(
            (self.params.nTimePeriods, self.params.nMicrotypes, self.params.nModesTotal))
        self.__subNetworkAverageSpeed = np.zeros(
            (self.params.nTimePeriods, self.params.nSubNetworks, self.params.nModesTotal))
        self.__subNetworkAccumulation = np.zeros(
            (self.params.nTimePeriods, self.params.nSubNetworks, self.params.nModesTotal))
        self.__subNetworkBlockedDistance = np.zeros(
            (self.params.nTimePeriods, self.params.nSubNetworks, self.params.nModesTotal))
        self.__subNetworkOperatingSpeed = np.zeros(
            (self.params.nTimePeriods, self.params.nSubNetworks, self.params.nModesTotal))
        self.__subNetworkVehicleSize = np.zeros((self.params.nSubNetworks, self.params.nModesTotal))
        self.__subNetworkLength = np.zeros((self.params.nSubNetworks, 1))
        self.__subNetworkScaledLength = np.zeros((self.params.nSubNetworks, 1))
        self.__subNetworkInstantaneousSpeed = np.zeros((self.params.nSubNetworks, self.params.nTimeSteps))
        self.__subNetworkInstantaneousAutoAccumulation = np.zeros((self.params.nSubNetworks, self.params.nTimeSteps))
        self.__instantaneousTime = np.arange(self.params.nTimeSteps) * self.params.timeStepInSeconds
        self.__transitionMatrixNetworkIdx = np.zeros(self.params.nSubNetworks, dtype=bool)
        self.__nonAutoModes = np.array([True] * self.params.nModesTotal)
        self.__nonAutoModes[self.scenarioData.modeToIdx['auto']] = False
        self.__MFDs = [[] for _ in range(self.params.nSubNetworks)]
        self.__freightProduction = np.zeros(
            (self.params.nTimePeriods, self.params.nMicrotypes, self.params.nFreightModes), dtype=float)

    def setModeStartCosts(self, mode, microtype, newCost, senior=None):
        data = self.__microtypeCosts[self.scenarioData.microtypeIdToIdx[microtype], :,
               self.scenarioData.modeToIdx[mode], 0]
        if senior is None:
            data.fill(newCost)
        else:
            isSenior = np.array([di.isSenior for di in self.scenarioData.diToIdx.keys()], dtype=bool)
            if senior:
                data[isSenior] = newCost
            else:
                data[~isSenior] = newCost

    def toStarts(self):
        return self.__toStarts

    def modeSplit(self, timePeriod=None):
        if timePeriod is None:
            return self.__modeSplit
        else:
            return self.__modeSplit[timePeriod, :, :, :]

    def tripRate(self, timePeriod=None):
        if timePeriod is None:
            return self.__tripRate
        else:
            return self.__tripRate[timePeriod, :, :]

    def updateTripRate(self, newTripRate, timePeriod=None):
        if timePeriod is None:
            np.copyto(self.__tripRate, newTripRate)
        else:
            np.copyto(self.__tripRate[timePeriod, :, :], newTripRate)

    def updateMicrotypeNetworkLength(self, microtypeID=None, newMultiplier=None):
        if newMultiplier is None:
            if microtypeID is None:
                for mIDx in self.scenarioData.microtypeIdToIdx.values():
                    microtypeMask = self.__subNetworkToMicrotype[mIDx, :]
                    subnetworkMask = microtypeMask & self.__transitionMatrixNetworkIdx
                    self.__subNetworkScaledLength[subnetworkMask] = self.__microtypeLengthMultiplier[:,
                                                                    mIDx].max() * self.__subNetworkLength[
                                                                        subnetworkMask]
            else:
                mIDx = self.scenarioData.microtypeIdToIdx[microtypeID]
                microtypeMask = self.__subNetworkToMicrotype[mIDx, :]
                subnetworkMask = microtypeMask & self.__transitionMatrixNetworkIdx
                self.__subNetworkScaledLength[subnetworkMask] = self.__microtypeLengthMultiplier[:, mIDx].max() * \
                                                                self.__subNetworkLength[subnetworkMask]
        else:
            mask = self.__subNetworkToMicrotype[self.scenarioData.microtypeIdToIdx[microtypeID], :]
            self.__subNetworkScaledLength[mask] = newMultiplier * self.__subNetworkLength[mask]
            self.__microtypeLengthMultiplier[:, self.scenarioData.microtypeIdToIdx[microtypeID]] = newMultiplier

    @property
    def t(self):
        return self.__instantaneousTime

    @property
    def v(self):
        return self.__subNetworkInstantaneousSpeed[self.__transitionMatrixNetworkIdx, :]

    @property
    def n(self):
        return self.__subNetworkInstantaneousAutoAccumulation[self.__transitionMatrixNetworkIdx, :]

    @property
    def utilities(self):
        return self.__utilities

    def getStartAndEndInd(self, timePeriodIdx):
        currentTimePeriodIndex = int(timePeriodIdx / self.params.nSubBins)
        currentTimePeriodDuration = self.scenarioData['timePeriods']['DurationInHours'][currentTimePeriodIndex]
        previousTimeInHours = self.scenarioData['timePeriods']['DurationInHours'][:currentTimePeriodIndex].sum()
        startTimeInHours = (timePeriodIdx % self.params.nSubBins) / float(
            self.params.nSubBins) * currentTimePeriodDuration + previousTimeInHours
        endTimeInHours = startTimeInHours + currentTimePeriodDuration / self.params.nSubBins
        return int(startTimeInHours * 3600 / self.params.timeStepInSeconds), \
               int(endTimeInHours * 3600 / self.params.timeStepInSeconds)

    def getSupply(self, timePeriodIdx=None):
        if timePeriodIdx is None:
            supply = dict()
            supply['demandData'] = self.__demandData
            supply['accessDistance'] = self.__accessDistance
            supply['microtypeSpeed'] = self.__microtypeSpeed
            supply['fleetSize'] = self.__fleetSize
            supply['microtypeMixedTrafficDistance'] = self.__microtypeMixedTrafficDistance
            supply['subNetworkAverageSpeed'] = self.__subNetworkAverageSpeed
            supply['subNetworkAccumulation'] = self.__subNetworkAccumulation
            supply['subNetworkBlockedDistance'] = self.__subNetworkBlockedDistance
            supply['subNetworkOperatingSpeed'] = self.__subNetworkOperatingSpeed
            supply['subNetworkVehicleSize'] = self.__subNetworkVehicleSize
            supply['subNetworkLength'] = self.__subNetworkLength
            supply['subNetworkScaledLength'] = self.__subNetworkScaledLength
            supply['subNetworkInstantaneousSpeed'] = self.__subNetworkInstantaneousSpeed
            supply['subNetworkInstantaneousAutoAccumulation'] = self.__subNetworkInstantaneousAutoAccumulation
            supply['subNetworkPreviousAutoAccumulation'] = np.zeros(self.params.nSubNetworks)
            supply['transitionMatrixNetworkIdx'] = self.__transitionMatrixNetworkIdx
            supply['nonAutoModes'] = self.__nonAutoModes
            supply['subNetworkToMicrotype'] = self.__subNetworkToMicrotype
            supply['microtypeCosts'] = self.__microtypeCosts
            supply['MFDs'] = self.__MFDs
            supply['freightProduction'] = self.__freightProduction
            supply['transitionMatrix'] = self.__transitionMatrix
            supply['transitionMatrices'] = self.__transitionMatrices
        else:
            startTimeStep, endTimeStep = self.getStartAndEndInd(timePeriodIdx)
            supply = dict()
            supply['demandData'] = self.__demandData[timePeriodIdx, :, :, :]
            supply['accessDistance'] = self.__accessDistance
            supply['microtypeSpeed'] = self.__microtypeSpeed[timePeriodIdx, :, :]
            supply['fleetSize'] = self.__fleetSize[timePeriodIdx, :, :]
            supply['microtypeMixedTrafficDistance'] = self.__microtypeMixedTrafficDistance[timePeriodIdx, :, :]
            supply['subNetworkAverageSpeed'] = self.__subNetworkAverageSpeed[timePeriodIdx, :, :]
            supply['subNetworkAccumulation'] = self.__subNetworkAccumulation[timePeriodIdx, :, :]
            supply['subNetworkBlockedDistance'] = self.__subNetworkBlockedDistance[timePeriodIdx, :, :]
            supply['subNetworkOperatingSpeed'] = self.__subNetworkOperatingSpeed[timePeriodIdx, :, :]
            supply['freightProduction'] = self.__freightProduction[timePeriodIdx, :, :]
            supply['subNetworkVehicleSize'] = self.__subNetworkVehicleSize
            supply['subNetworkLength'] = self.__subNetworkLength
            supply['subNetworkScaledLength'] = self.__subNetworkScaledLength
            supply['subNetworkInstantaneousSpeed'] = self.__subNetworkInstantaneousSpeed[:, startTimeStep:endTimeStep]
            supply['subNetworkInstantaneousAutoAccumulation'] = self.__subNetworkInstantaneousAutoAccumulation[
                                                                :, startTimeStep:endTimeStep]
            if startTimeStep == 0:
                supply['subNetworkPreviousAutoAccumulation'] = np.zeros(self.params.nSubNetworks)
            else:
                supply['subNetworkPreviousAutoAccumulation'] = self.__subNetworkInstantaneousAutoAccumulation[:,
                                                               startTimeStep - 1]
            supply['transitionMatrixNetworkIdx'] = self.__transitionMatrixNetworkIdx
            supply['nonAutoModes'] = self.__nonAutoModes
            supply['subNetworkToMicrotype'] = self.__subNetworkToMicrotype
            supply['microtypeCosts'] = self.__microtypeCosts
            supply['MFDs'] = self.__MFDs
            supply['transitionMatrix'] = self.__transitionMatrix[timePeriodIdx, :, :]
            supply['transitionMatrices'] = self.__transitionMatrices
        return supply

    def getDemand(self, timePeriodIdx=None):
        demand = dict()
        if timePeriodIdx is None:
            demand['demandData'] = self.__demandData
            demand['modeSplit'] = self.__modeSplit
            demand['tripRate'] = self.__tripRate
            demand['fleetSize'] = self.__fleetSize
            demand['toStarts'] = self.__toStarts
            demand['toEnds'] = self.__toEnds
            demand['toThroughDistance'] = self.__toThroughDistance
            demand['toDistanceByOrigin'] = self.__toDistanceByOrigin
            demand['utilities'] = self.__utilities
            demand['choiceCharacteristics'] = self.__choiceCharacteristics
            demand['choiceParameters'] = self.__choiceParameters
            demand['choiceParametersFixed'] = self.__choiceParametersFixed
            demand['toTransitLayer'] = self.__toTransitLayer
            demand['transitLayerUtility'] = self.__transitLayerUtility
        else:
            demand['demandData'] = self.__demandData[timePeriodIdx, :, :, :]
            demand['modeSplit'] = self.__modeSplit[timePeriodIdx, :, :, :]
            demand['tripRate'] = self.__tripRate[timePeriodIdx, :, :]
            demand['fleetSize'] = self.__fleetSize[timePeriodIdx, :, :]
            demand['toStarts'] = self.__toStarts
            demand['toEnds'] = self.__toEnds
            demand['toThroughDistance'] = self.__toThroughDistance
            demand['toDistanceByOrigin'] = self.__toDistanceByOrigin
            demand['utilities'] = self.__utilities[timePeriodIdx, :, :, :]
            demand['choiceCharacteristics'] = self.__choiceCharacteristics[timePeriodIdx, :, :, :, :]
            demand['choiceParameters'] = self.__choiceParameters
            demand['choiceParametersFixed'] = self.__choiceParametersFixed
            demand['toTransitLayer'] = self.__toTransitLayer
            demand['transitLayerUtility'] = self.__transitLayerUtility
        return demand

    def getInvariants(self):
        fixedData = dict()
        fixedData['subNetworkVehicleSize'] = self.__subNetworkVehicleSize
        fixedData['subNetworkLength'] = self.__subNetworkLength
        fixedData['subNetworkScaledLength'] = self.__subNetworkScaledLength
        fixedData['toStarts'] = self.__toStarts
        fixedData['toEnds'] = self.__toEnds
        fixedData['microtypeCosts'] = self.__microtypeCosts
        fixedData['toThroughDistance'] = self.__toThroughDistance
        fixedData['toDistanceByOrigin'] = self.__toDistanceByOrigin
        fixedData['toTripPurpose'] = self.__toTripPurpose
        fixedData['toHomeMicrotype'] = self.__toHomeMicrotype
        fixedData['toODI'] = self.__toODI
        fixedData['choiceParameters'] = self.__choiceParameters
        fixedData['choiceParametersFixed'] = self.__choiceParametersFixed
        fixedData['transitionMatrixNetworkIdx'] = self.__transitionMatrixNetworkIdx
        fixedData['nonAutoModes'] = self.__nonAutoModes
        fixedData['toTransitLayer'] = self.__toTransitLayer
        fixedData['transitLayerUtility'] = self.__transitLayerUtility
        fixedData['subNetworkToMicrotype'] = self.__subNetworkToMicrotype
        fixedData['accessDistance'] = self.__accessDistance
        fixedData['activityDensity'] = self.__activityDensity
        return fixedData

    def updateNetworkLength(self, networkId, newLength):
        idx = self.scenarioData.subNetworkIdToIdx[networkId]
        self.__subNetworkLength[idx] = newLength
        self.__subNetworkScaledLength[idx] = newLength * \
                                             self.__microtypeLengthMultiplier[:,
                                             self.__subNetworkToMicrotype[:, idx]].max(
                                                 axis=0)[0]
