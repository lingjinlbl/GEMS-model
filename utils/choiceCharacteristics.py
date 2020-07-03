from utils.OD import TripCollection, OriginDestination, ODindex
from utils.microtype import MicrotypeCollection
from utils.misc import DistanceBins


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


class CollectedChoiceCharacteristics:
    def __init__(self):
        self.__choiceCharacteristics = dict()

    def __setitem__(self, key: ODindex, value: ModalChoiceCharacteristics):
        self.__choiceCharacteristics[key] = value

    def __getitem__(self, item: ODindex) -> ModalChoiceCharacteristics:
        return self.__choiceCharacteristics[item]

    def initializeChoiceCharacteristics(self, originDestination: OriginDestination, trips: TripCollection,
                                        microtypes: MicrotypeCollection, distanceBins: DistanceBins):
        for odIndex, trip in trips:
            common_modes = []
            for microtypeID, allocation in trip.allocation:
                if allocation > 0:
                    common_modes.append(microtypes[microtypeID].mode_names)
            modes = set.intersection(*common_modes)
            self[odIndex] = ModalChoiceCharacteristics(modes)
