import utils.IO as io
import utils.OD as od
from utils.IO import CollectedModeCharacteristics

from utils.microtype import Microtype

network_params_default = io.Network(0.068, 15.42, 1.88, 0.145, 0.177, 1000, 50)
bus_params_default = io.BusParams(mean_trip_distance=1000, road_network_fraction=500, relative_length=3.0,
                                  fixed_density=95. / 100., min_stop_time=15., stop_spacing=1. / 250.,
                                  passenger_wait=5.)

car_params_default = io.ModeParams(mean_trip_distance=1000, relative_length=1.0)

modeCharacteristics: CollectedModeCharacteristics = io.CollectedModeCharacteristics()
modeCharacteristics['car'] = io.ModeCharacteristics('car', car_params_default, demand=0.)#70 / (10 * 60)
modeCharacteristics['bus'] = io.ModeCharacteristics('bus', bus_params_default, demand=0.)#17 / (10 * 60)

m = Microtype(network_params_default, modeCharacteristics)

ODtest = od.OD(m, m)

DUtest = od.DemandUnit(distance=1000, demand=0.1, allocation=od.Allocation({m: 1.0}),
                       mode_split=od.ModeSplit({'car': 0.75, 'bus': 0.25}))

DUtest2 = od.DemandUnit(distance=2000, demand=0.1, allocation=od.Allocation({m: 1.0}),
                       mode_split=od.ModeSplit({'car': 0.75, 'bus': 0.25}))
ODtest.append(DUtest)
ODtest.append(DUtest2)