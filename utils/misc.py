import pandas as pd


class TimePeriods:
    def __init__(self):
        self.__timePeriods = dict()

    def __setitem__(self, key: str, value: float):
        self.__timePeriods[key] = value

    def __getitem__(self, item) -> float:
        return self.__timePeriods[item]

    def __iter__(self):
        return iter(self.__timePeriods.items())

    def importTimePeriods(self, df: pd.DataFrame):
        for row in df.itertuples():
            self[row.TimePeriodID] = row.DurationInHours


class DistanceBins:
    def __init__(self):
        self.__distanceBins = dict()

    def __setitem__(self, key: str, value: float):
        self.__distanceBins[key] = value

    def __getitem__(self, item) -> float:
        return float(self.__distanceBins[item])

    def importDistanceBins(self, df: pd.DataFrame):
        for row in df.itertuples():
            self[row.DistanceBinID] = row.MeanDistanceInMiles
