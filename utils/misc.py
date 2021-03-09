import pandas as pd
from collections import OrderedDict


class TimePeriods:
    """
    Collection of time period of day and length of said periods.
    """
    def __init__(self):
        self.__timePeriods = OrderedDict()

    def __setitem__(self, key: str, value: float):
        self.__timePeriods[key] = value

    def __getitem__(self, item) -> float:
        return self.__timePeriods[item]

    def __iter__(self):
        return iter(self.__timePeriods.items())

    def importTimePeriods(self, df: pd.DataFrame):
        for row in df.itertuples():
            self[row.TimePeriodID] = row.DurationInHours
        print("|  Loaded ", len(df), " time periods")

    def __contains__(self, item):
        if item in self.__timePeriods:
            return True
        else:
            return False


class DistanceBins:
    """
    Collection of distances of trips.
    """
    def __init__(self):
        self.__distanceBins = dict()

    def __setitem__(self, key: str, value: float):
        self.__distanceBins[key] = value

    def __getitem__(self, item) -> float:
        return float(self.__distanceBins[item])

    def importDistanceBins(self, df: pd.DataFrame):
        for row in df.itertuples():
            self[row.DistanceBinID] = row.MeanDistanceInMiles
        print("|  Loaded ", len(df), " distance bins")
