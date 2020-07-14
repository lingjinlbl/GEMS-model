from .OD import TripCollection, ODindex
from .microtype import MicrotypeCollection
from .misc import DistanceBins


class ChoiceCharacteristics:
    def __init__(self, travel_time=0., cost=0., wait_time=0.):
        self.travel_time = travel_time
        self.cost = cost
        self.wait_time = wait_time

    def __add__(self, other):
        if isinstance(other, ChoiceCharacteristics):
            self.travel_time += other.travel_time
            self.cost += other.cost
            self.wait_time += other.wait_time
            return self
        else:
            print('TOUGH LUCK, BUDDY')
            return self

    def __iadd__(self, other):
        if isinstance(other, ChoiceCharacteristics):
            self.travel_time += other.travel_time
            self.cost += other.cost
            self.wait_time += other.wait_time
            return self
        else:
            print('TOUGH LUCK, BUDDY')
            return self


class ModalChoiceCharacteristics:
    def __init__(self, modes):
        self.__modalChoiceCharacteristics = dict()
        for mode in modes:
            self.__modalChoiceCharacteristics[mode] = ChoiceCharacteristics()

    def __getitem__(self, item: str) -> ChoiceCharacteristics:
        return self.__modalChoiceCharacteristics[item]

    def __setitem__(self, key: str, value: ChoiceCharacteristics):
        self.__modalChoiceCharacteristics[key] = value

    def modes(self):
        return list(self.__modalChoiceCharacteristics.keys())

    def reset(self):
        for mode in self.modes():
            self[mode] = ChoiceCharacteristics()


class CollectedChoiceCharacteristics:
    def __init__(self):
        self.__choiceCharacteristics = dict()
        self.__distanceBins = DistanceBins()

    def __setitem__(self, key: ODindex, value: ModalChoiceCharacteristics):
        self.__choiceCharacteristics[key] = value

    def __getitem__(self, item: ODindex) -> ModalChoiceCharacteristics:
        return self.__choiceCharacteristics[item]

    def initializeChoiceCharacteristics(self, trips: TripCollection,
                                        microtypes: MicrotypeCollection, distanceBins: DistanceBins):
        self.__distanceBins = distanceBins
        for odIndex, trip in trips:
            common_modes = []
            for microtypeID, allocation in trip.allocation:
                if allocation > 0:
                    common_modes.append(microtypes[microtypeID].mode_names)
            modes = set.intersection(*common_modes)
            self[odIndex] = ModalChoiceCharacteristics(modes)

    def resetChoiceCharacteristics(self):
        for mcc in self.__choiceCharacteristics.values():
            mcc.reset()

    def updateChoiceCharacteristics(self, microtypes: MicrotypeCollection, trips: TripCollection):
        self.resetChoiceCharacteristics()
        for odIndex, trip in trips:
            for mode in microtypes[odIndex.o].mode_names:
                self[odIndex][mode] += ChoiceCharacteristics(*microtypes[odIndex.o].getStartTimeCostWait(mode))
            for mode in microtypes[odIndex.d].mode_names:
                self[odIndex][mode] += ChoiceCharacteristics(*microtypes[odIndex.d].getEndTimeCostWait(mode))
            for microtypeID, allocation in trip.allocation:
                if allocation > 0:
                    for mode in microtypes[microtypeID].mode_names:
                        self[odIndex][mode] += ChoiceCharacteristics(
                            *microtypes[microtypeID].getThroughTimeCostWait(mode, self.__distanceBins[
                                odIndex.distBin] * allocation))
