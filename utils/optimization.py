from model import Model
import numpy as np
import pandas as pd
import os
from scipy.optimize import root, minimize, Bounds, shgo, least_squares


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

    def loadData(self, path):
        speed = pd.read_csv(os.path.join(path, "calibration", "avg_speed_from_HERE.csv"),
                            index_col=['microtype', 'hour'])
        modeSplit = pd.read_csv(os.path.join(path, "calibration", "NHTS_mode_split.csv"),
                                index_col=['origin microtype', 'mode'])
        travelTime = pd.read_csv(os.path.join(path, "calibration", "NHTS_trip_travel_time.csv"),
                                 index_col=['origin microtype', 'mode'])
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
                self.values[idx] = row["speed (mph)"]
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
                    self.values[idx] = val
                    idx += 1

    def getError(self, modeSplitData, speedData, utilityData) -> np.ndarray:
        startIdx = 0
        yHat = np.zeros_like(self.values)
        if len(self.__modeSplitData) > 0:
            tripsByModeAndOrigin = modeSplitData.stack(level=0)['Trips'].groupby(level=['originMicrotype', 'mode']).agg(
                sum).unstack(level=0)
            modeSplit = (tripsByModeAndOrigin / tripsByModeAndOrigin.sum(axis=0)).unstack()
            yHat[:len(modeSplit)] = modeSplit.loc[self.__modeSplitData.index].values
            startIdx += len(modeSplit)
        if len(self.__speedData) > 0:
            autoSpeedByMicrotypeAndTimePeriod = speedData.stack(level=0)['Speed'].unstack(level=1)['auto']
            yHat[startIdx:(startIdx + len(autoSpeedByMicrotypeAndTimePeriod))] = autoSpeedByMicrotypeAndTimePeriod.loc[
                self.__speedData.index].values
            startIdx += len(autoSpeedByMicrotypeAndTimePeriod)
        if len(self.__travelTimeData) > 0:
            totalTrips = modeSplitData.loc[utilityData.index].swaplevel(axis=1)['Trips']
            timeData = utilityData.swaplevel(axis=1, i=1, j=0)['Value'].swaplevel(axis=1)
            travelTimePerTrip = timeData['access_time'] + timeData['travel_time'] + timeData['wait_time']
            totalDistance = (timeData['distance'] * totalTrips).sum(axis=1).groupby(
                level=['originMicrotype', 'mode']).agg(sum)
            totalTime = (travelTimePerTrip * totalTrips).sum(axis=1).groupby(level=['originMicrotype', 'mode']).agg(sum)
            for column in self.__columnsFromTravelTime:
                if column == "avg_speed (mph)":
                    out = totalDistance / totalTime
                elif column == "total_travel_time":
                    out = totalTime
                elif column == "total_distance":
                    out = totalDistance
                else:
                    NotImplementedError("Cannot use column {0} for validation".format(column))
                out = out.loc[self.__travelTimeData.index]
                yHat[startIdx:(startIdx + len(out))] = out.values
                startIdx += len(out)
        return yHat


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

    @property
    def x0(self):
        return self.defaults / self.scaling

    @staticmethod
    def __getDefaults(variable) -> (float, float, float):
        changeType, _ = variable
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

    def modifyModelInPlace(self, model: Model, x: np.ndarray):
        scaledX = x * self.scaling
        for idx, var in self.idxToVariable.items():
            model.interact.modifyModel(var, scaledX[idx])


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
        return error

    def calibrate(self, method=None):
        return least_squares(self.f, self.optimizationVariables.x0,
                             bounds=(self.optimizationVariables.minimums / self.optimizationVariables.scaling,
                                     self.optimizationVariables.maximums / self.optimizationVariables.scaling),
                             method=method, verbose=2, diff_step=0.0001, xtol=1e-9)


if __name__ == "__main__":
    model = Model("../input-data", 1, False)

    calibrationVariableNames = [('accessDistanceMultiplier', ('A', 'Bus')),
                                ('accessDistanceMultiplier', ('B', 'Bus')),
                                ('accessDistanceMultiplier', ('C', 'Bus')),
                                ('accessDistanceMultiplier', ('D', 'Bus'))]

    calibrationVariables = OptimizationVariables(calibrationVariableNames)
    calibrator = Calibrator(model, calibrationVariables)
    result = calibrator.calibrate('trf')
    print('done')
