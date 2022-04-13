from model import Model
import numpy as np
import pandas as pd
import os
from scipy.optimize import root, minimize, Bounds, shgo, least_squares
from skopt import gp_minimize


class CalibrationValues:
    def __init__(self, speed=pd.DataFrame(), modeSplit=pd.DataFrame(), travelTime=pd.DataFrame(),
                 columnsFromTravelTime=('avg_speed (mph)',)):
        self.__speedData = speed
        self.__modeSplitData = modeSplit
        self.__travelTimeData = travelTime
        self.__columnsFromTravelTime = columnsFromTravelTime
        self.__idxToValue = dict()
        self.values = np.zeros(
            len(speed) + len(modeSplit) + len(travelTime) * len(columnsFromTravelTime))
        self.__speedScaling = 20.0

    def loadData(self, path):
        speed = pd.read_csv(os.path.join(path, "calibration", "avg_speed_from_HERE.csv"),
                            dtype={"microtype": str}).set_index(['microtype', 'hour'])
        modeSplit = pd.read_csv(os.path.join(path, "calibration", "NHTS_mode_split.csv"),
                                dtype={"origin microtype": str}).set_index(['origin microtype', 'mode'])
        travelTime = pd.read_csv(os.path.join(path, "calibration", "NHTS_trip_travel_time.csv"),
                                 dtype={"origin microtype": str}).set_index(['origin microtype', 'mode'])
        self.__speedData = speed
        self.__modeSplitData = modeSplit
        self.__travelTimeData = travelTime
        self.values = np.zeros(len(speed) + len(modeSplit) + len(travelTime) * len(self.__columnsFromTravelTime))
        self.__readFiles()

    def __readFiles(self):
        idx = 0
        if len(self.__speedData) > 0:
            for (microtype, hour), row in self.__speedData.iterrows():
                key = (microtype, "auto", "hourlySpeed", hour)
                self.__idxToValue[idx] = key
                self.values[idx] = row["speed (mph)"] / self.__speedScaling
                idx += 1
        if len(self.__modeSplitData) > 0:
            for (microtype, mode), row in self.__modeSplitData.iterrows():
                key = (microtype, mode, "modeSplit", -1)
                self.__idxToValue[idx] = key
                self.values[idx] = row["fraction"]
                idx += 1
        if len(self.__travelTimeData) > 0:
            for col in self.__columnsFromTravelTime:
                for (microtype, mode), val in self.__travelTimeData[col].iteritems():
                    key = (microtype, mode, col, -1)
                    self.__idxToValue[idx] = key
                    self.values[idx] = val / self.__speedScaling
                    idx += 1

    def getError(self, modeSplitData, speedData, utilityData) -> np.ndarray:
        startIdx = 0
        yHat = np.zeros_like(self.values)
        if len(self.__speedData) > 0:
            autoSpeedByMicrotypeAndTimePeriod = speedData.stack(level=0)['Speed'].unstack(level=1)['auto'] * 3600 / 1609
            newIndex = self.__speedData.index
            yHat[startIdx:(startIdx + len(autoSpeedByMicrotypeAndTimePeriod))] = autoSpeedByMicrotypeAndTimePeriod.loc[
                                                                                     newIndex].values / self.__speedScaling
            startIdx += len(autoSpeedByMicrotypeAndTimePeriod)
        if len(self.__modeSplitData) > 0:
            tripsByModeAndOrigin = modeSplitData.stack(level=0)['Trips'].groupby(level=['originMicrotype', 'mode']).agg(
                sum).unstack(level=0)
            modeSplit = (tripsByModeAndOrigin / tripsByModeAndOrigin.sum(axis=0)).unstack()
            filteredModeSplit = modeSplit.loc[self.__modeSplitData.index].values
            yHat[startIdx:(startIdx + len(modeSplit))] = filteredModeSplit
            startIdx += len(modeSplit)
        if len(self.__travelTimeData) > 0:
            totalTrips = modeSplitData.loc[utilityData.index].swaplevel(axis=1)['Trips']
            timeData = utilityData.swaplevel(axis=1, i=1, j=0)['Value'].swaplevel(axis=1)
            travelTimePerTrip = timeData['access_time'] + timeData['travel_time'] + timeData['wait_time']
            totalDistance = (timeData['distance'] * totalTrips).sum(axis=1).groupby(
                level=['originMicrotype', 'mode']).agg(sum)
            totalTime = (travelTimePerTrip * totalTrips).sum(axis=1).groupby(level=['originMicrotype', 'mode']).agg(sum)
            for column in self.__columnsFromTravelTime:
                if column == "avg_speed (mph)":
                    out = totalDistance / totalTime / self.__speedScaling
                elif column == "total_travel_time":
                    out = totalTime
                elif column == "total_distance":
                    out = totalDistance
                else:
                    NotImplementedError("Cannot use column {0} for validation".format(column))
                out = out.loc[self.__travelTimeData.index]
                yHat[startIdx:(startIdx + len(out))] = out.values
                startIdx += len(out)
        return yHat - self.values

    def errorToPandas(self, error: np.ndarray):
        return pd.Series(error, index=pd.MultiIndex.from_tuples(list(self.__idxToValue.values())))


class OptimizationVariables:
    def __init__(self, variables: list, minimums=None, maximums=None, defaults=None):
        self.__variables = variables
        self.maximums = np.zeros(len(variables))
        self.minimums = np.zeros(len(variables))
        self.defaults = np.zeros(len(variables))
        self.scaling = np.ones(len(variables))
        self.idxToVariable = dict()
        if minimums is None:
            self.__minimums = dict()
        else:
            self.__minimums = minimums
        if maximums is None:
            self.__maximums = dict()
        else:
            self.__maximums = maximums
        if defaults is None:
            self.__defaults = dict()
        else:
            self.__defaults = defaults
        self.__defineRanges()

    def __defineRanges(self):
        for idx, variable in enumerate(self.__variables):
            self.idxToVariable[idx] = variable
            minimum, maximum, default = self.__getDefaults(variable)
            if variable in self.__minimums:
                minimum = self.__minimums[variable]
            if variable in self.__maximums:
                maximum = self.__maximums[variable]
            if variable in self.__defaults:
                default = self.__defaults[variable]
            self.minimums[idx] = minimum
            self.maximums[idx] = maximum
            self.defaults[idx] = default
            self.scaling[idx] = maximum - minimum

    def defineDefaultsFromModel(self, model: Model):
        for idx, variable in self.idxToVariable.items():
            self.defaults[idx] = model.interact.getModelState(variable)

    @property
    def bounds(self):
        return (self.minimums - self.defaults) / self.scaling, (self.maximums - self.defaults) / self.scaling

    @property
    def x0(self):
        return np.zeros_like(self.defaults)

    @staticmethod
    def __getDefaults(variable) -> (float, float, float):
        changeType, (microtype, mode) = variable
        if changeType == "dedicated":
            return 0.0, 0.75, 0.0
        elif changeType == 'headway':
            return 60., 1800., 360.
        elif (changeType == "fare") | (changeType == "fareSenior"):
            return 0.0, 15.0, 2.5
        elif changeType == "coverage":
            return 0.01, 1.0, 0.25
        elif changeType == "vMax":
            return 10.0, 30.0, 17.0
        elif changeType == "densityMax":
            return 0.1, 1.0, 0.4
        elif changeType == "a":
            return -1500.0, -200.0, -250.0
        elif changeType == "b":
            return 0.01666667, 0.1, 0.05
        elif changeType == "accessDistanceMultiplier":
            return 0.0, 2.0, 0.3
        elif changeType == "minStopTime":
            return 0.0, 60.0, 10.0
        elif changeType == "passengerWait":
            return 0.0, 15.0, 5.0
        elif changeType == "stopSpacing":
            return 100.0, 3200.0, 400.0
        elif changeType == "modeSpeedMPH":
            if mode == "Walk":
                return 1.5, 4.0, 3.0
            elif mode == "Bike":
                return 4.0, 10.0, 6.0
            elif mode == "Rail":
                return 10.0, 60.0, 25.0

    def modifyModelInPlace(self, model: Model, x: np.ndarray):
        scaledX = x * self.scaling + self.defaults
        for idx, var in self.idxToVariable.items():
            model.interact.modifyModel(var, scaledX[idx])

    def toPandas(self, x):
        scaledX = x * self.scaling + self.defaults
        return pd.Series(scaledX, index=pd.MultiIndex.from_tuples(self.__variables))


class Calibrator:
    def __init__(self, model: Model, optimizationVariables: OptimizationVariables):
        self.model = model
        self.optimizationVariables = optimizationVariables
        self.calibrationVariables = CalibrationValues()
        self.calibrationVariables.loadData(path=model.path)

    def f(self, x: np.ndarray) -> np.ndarray:
        self.optimizationVariables.modifyModelInPlace(self.model, x)
        self.model.collectAllCharacteristics()
        modeSplitData, speedData, utilityData, _ = self.model.toPandas()
        error = self.calibrationVariables.getError(modeSplitData, speedData, utilityData)
        error[np.isnan(error)] = 5
        return error

    def calibrate(self, method=None):
        return least_squares(self.f, self.optimizationVariables.x0,
                             bounds=self.optimizationVariables.bounds,
                             method=method, verbose=2, diff_step=0.01, xtol=1e-5)


if __name__ == "__main__":
    model = Model("../input-data-california-A", 1, False)

    # calibrationVariableNames = [('accessDistanceMultiplier', ('A', 'Bus')),
    #                             ('accessDistanceMultiplier', ('B', 'Bus')),
    #                             ('accessDistanceMultiplier', ('C', 'Bus')),
    #                             ('accessDistanceMultiplier', ('D', 'Bus')),
    #                             ('minStopTime', ('A', 'Bus')),
    #                             ('minStopTime', ('B', 'Bus')),
    #                             ('minStopTime', ('C', 'Bus')),
    #                             ('minStopTime', ('D', 'Bus')),
    #                             ('modeSpeedMPH', ('A', 'Walk')),
    #                             ('modeSpeedMPH', ('B', 'Walk')),
    #                             ('modeSpeedMPH', ('C', 'Walk')),
    #                             ('modeSpeedMPH', ('D', 'Walk')),
    #                             ('modeSpeedMPH', ('A', 'Bike')),
    #                             ('modeSpeedMPH', ('B', 'Bike')),
    #                             ('modeSpeedMPH', ('C', 'Bike')),
    #                             ('modeSpeedMPH', ('D', 'Bike')),
    #                             ('modeSpeedMPH', ('A', 'Rail')),
    #                             ('modeSpeedMPH', ('B', 'Rail')),
    #                             ('modeSpeedMPH', ('C', 'Rail')),
    #                             ('modeSpeedMPH', ('D', 'Rail')),
    #                             ('passengerWait', ('A', 'Bus')),
    #                             ('passengerWait', ('B', 'Bus')),
    #                             ('passengerWait', ('C', 'Bus')),
    #                             ('passengerWait', ('D', 'Bus'))]

    calibrationVariableNames = [('accessDistanceMultiplier', ('1', 'Bus')),
                                ('accessDistanceMultiplier', ('2', 'Bus')),
                                ('accessDistanceMultiplier', ('3', 'Bus')),
                                ('accessDistanceMultiplier', ('4', 'Bus')),
                                ('accessDistanceMultiplier', ('5', 'Bus')),
                                ('accessDistanceMultiplier', ('6', 'Bus')),
                                ('minStopTime', ('1', 'Bus')),
                                ('minStopTime', ('2', 'Bus')),
                                ('minStopTime', ('3', 'Bus')),
                                ('minStopTime', ('4', 'Bus')),
                                ('minStopTime', ('5', 'Bus')),
                                ('minStopTime', ('6', 'Bus')),
                                ('modeSpeedMPH', ('1', 'Walk')),
                                ('modeSpeedMPH', ('2', 'Walk')),
                                ('modeSpeedMPH', ('3', 'Walk')),
                                ('modeSpeedMPH', ('4', 'Walk')),
                                ('modeSpeedMPH', ('5', 'Walk')),
                                ('modeSpeedMPH', ('6', 'Walk')),
                                ('modeSpeedMPH', ('1', 'Bike')),
                                ('modeSpeedMPH', ('2', 'Bike')),
                                ('modeSpeedMPH', ('3', 'Bike')),
                                ('modeSpeedMPH', ('4', 'Bike')),
                                ('modeSpeedMPH', ('5', 'Bike')),
                                ('modeSpeedMPH', ('6', 'Bike')),
                                ('modeSpeedMPH', ('1', 'Rail')),
                                ('modeSpeedMPH', ('2', 'Rail')),
                                ('modeSpeedMPH', ('3', 'Rail')),
                                ('modeSpeedMPH', ('4', 'Rail')),
                                ('modeSpeedMPH', ('5', 'Rail')),
                                ('modeSpeedMPH', ('6', 'Rail')),
                                ('passengerWait', ('1', 'Bus')),
                                ('passengerWait', ('2', 'Bus')),
                                ('passengerWait', ('3', 'Bus')),
                                ('passengerWait', ('4', 'Bus')),
                                ('passengerWait', ('5', 'Bus')),
                                ('passengerWait', ('6', 'Bus'))]

    calibrationVariables = OptimizationVariables(calibrationVariableNames)
    calibrator = Calibrator(model, calibrationVariables)
    result = calibrator.calibrate('trf')
    final = calibrationVariables.toPandas(result.x).unstack()
    print(final)
    calibrator.optimizationVariables.toPandas(result.x).to_csv('../calibration-outputs.csv')
    print('done')
