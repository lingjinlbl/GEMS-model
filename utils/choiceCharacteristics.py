# from .microtype import MicrotypeCollection
from .misc import DistanceBins


class ChoiceCharacteristics:
    def __init__(self, travel_time=0., cost=0., wait_time=0., access_time=0, protected_distance=0, distance=0):
        self.travel_time = travel_time
        self.cost = cost
        self.wait_time = wait_time
        self.access_time = access_time
        self.protected_distance = protected_distance
        self.distance = distance

    def __add__(self, other):
        if isinstance(other, ChoiceCharacteristics):
            self.travel_time += other.travel_time
            self.cost += other.cost
            self.wait_time += other.wait_time
            self.access_time += other.access_time
            self.protected_distance += other.protected_distance
            self.distance += other.distance
            return self
        else:
            print('TOUGH LUCK, BUDDY')
            return self

    def __iadd__(self, other):
        if isinstance(other, ChoiceCharacteristics):
            self.travel_time += other.travel_time
            self.cost += other.cost
            self.wait_time += other.wait_time
            self.access_time += other.access_time
            self.protected_distance += other.protected_distance
            self.distance += other.distance
            return self
        else:
            print('TOUGH LUCK, BUDDY')
            return self


class ModalChoiceCharacteristics:
    def __init__(self, modes, distanceInMiles=0.0):
        self.__modalChoiceCharacteristics = dict()
        self.distanceInMiles = distanceInMiles
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

    def __contains__(self, item):
        return item in self.__modalChoiceCharacteristics


class CollectedChoiceCharacteristics:
    def __init__(self):
        self.__choiceCharacteristics = dict()
        self.__distanceBins = DistanceBins()

    def __setitem__(self, key, value: ModalChoiceCharacteristics):
        self.__choiceCharacteristics[key] = value

    def __getitem__(self, item) -> ModalChoiceCharacteristics:
        return self.__choiceCharacteristics[item]

    def initializeChoiceCharacteristics(self, trips,
                                        microtypes, distanceBins: DistanceBins):
        self.__distanceBins = distanceBins
        for odIndex, trip in trips:
            common_modes = [microtypes[odIndex.o].mode_names, microtypes[odIndex.d].mode_names]
            # common_modes = []
            # for microtypeID, allocation in trip.allocation:
            #     if allocation > 0:
            #         common_modes.append(microtypes[microtypeID].mode_names)
            modes = set.intersection(*common_modes)
            self[odIndex] = ModalChoiceCharacteristics(modes, distanceBins[odIndex.distBin])

    def resetChoiceCharacteristics(self):
        for mcc in self.__choiceCharacteristics.values():
            mcc.reset()

    def updateChoiceCharacteristics(self, microtypes, trips):
        self.resetChoiceCharacteristics()
        for odIndex, trip in trips:
            common_modes = [microtypes[odIndex.o].mode_names, microtypes[odIndex.d].mode_names]
            modes = set.intersection(*common_modes)
            for mode in modes:
                self[odIndex][mode] += microtypes[odIndex.o].getStartTimeCostWait(mode)
                self[odIndex][mode] += microtypes[odIndex.d].getEndTimeCostWait(mode)
                newAllocation = filterAllocation(mode, trip.allocation, microtypes)
                for microtypeID, allocation in newAllocation.items():
                    self[odIndex][mode] += microtypes[microtypeID].getThroughTimeCostWait(mode, self.__distanceBins[
                        odIndex.distBin] * allocation)
                # assert self[odIndex][mode].distance == self[odIndex].distanceInMiles


def filterAllocation(mode: str, inputAllocation, microtypes):
    through_microtypes = []
    allocation = []
    tot = 0.0
    for m, a in inputAllocation:
        if (a > 0) & (mode in microtypes[m].mode_names):
            through_microtypes.append(m)
            allocation.append(a)
            tot += a
    # allocation = np.array(allocation) / tot
    # allocation /= np.sum(allocation)
    return {m: a / tot for m, a in zip(through_microtypes, allocation)}  # dict(zip(through_microtypes, allocation))
