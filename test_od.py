import utils.IO as io
import utils.OD as od

from utils.microtype import Microtype, CollectedModeCharacteristics, ModeCharacteristics
from utils.supply import ModeParams, BusParams
from utils.Network import Network
from utils.geotype import Geotype

network_params = Network(0.068, 15.42, 1.88, 0.145, 0.177, 1000, 50)
bus_params_default = BusParams(road_network_fraction=500, relative_length=3.0,
                               fixed_density=95. / 100., min_stop_time=15., stop_spacing=1. / 250.,
                               passenger_wait=5.)

car_params_default = ModeParams(relative_length=1.0)

modeCharacteristics = CollectedModeCharacteristics()
modeCharacteristics['car'] = ModeCharacteristics('car', car_params_default)
modeCharacteristics['bus'] = ModeCharacteristics('bus', bus_params_default)

m = Microtype(network_params, modeCharacteristics)
m.setModeDemand('car', 70 / (10 * 60), 1000.0)
m.setModeDemand('bus', 10 / (10 * 60), 1000.0)

DUtest = od.DemandUnit(distance=1000, demand=0.1, allocation=od.Allocation({m: 1.0}),
                       mode_split=od.ModeSplit({'car': 0.75, 'bus': 0.25}))

distbins = {0: 1000.0, 1: 2000}
demandbydistbin = {0: 40 / 600., 1: 40 / 600.}
all_car = od.ModeSplit({'car': 1.0})
split = od.ModeSplit({'car': 0.8, 'bus': 0.2})

g = Geotype(distbins=distbins)
g += m
g.appendDemandData(od.ODindex(m, m, 0), od.DemandUnit(distbins[0], demandbydistbin[0], od.Allocation({m: 1.0}), split))
g.appendDemandData(od.ODindex(m, m, 1),
                   od.DemandUnit(distbins[1], demandbydistbin[1], od.Allocation({m: 1.0}), all_car))
