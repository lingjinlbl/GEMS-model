import utils.OD as od
import numpy as np
from utils.microtype import Microtype, CollectedModeCharacteristics, ModeCharacteristics, Costs
from utils.supply import ModeParams, BusParams
from utils.network import Network
from utils.geotype import Geotype, getModeSplit
import matplotlib.pyplot as plt

costs = {'car': Costs(0.0003778, 0., 1.0, 1.0), 'bus': Costs(0., 2.5, 0., 1.)}

network_params = Network(0.068, 15.42, 1.88, 0.145, 0.177, 1000, 50)
bus_params_default = BusParams(road_network_fraction=500, relative_length=3.0,
                               fixed_density=95. / 100., min_stop_time=15., stop_spacing=1. / 250.,
                               passenger_wait=5.)

car_params_default = ModeParams(relative_length=1.0)

modeCharacteristics = ModeCharacteristics('car', car_params_default) + ModeCharacteristics('bus', bus_params_default)

m = Microtype(network_params, modeCharacteristics, costs)

distbins = {0: 1000.0, 1: 2000}
demandbydistbin = {0: 30 / 600., 1: 25 / 600.}
modesplitbydistbin = {0: od.ModeSplit({'car': 0.75, 'bus': 0.25}), 1: od.ModeSplit({'car': 0.75, 'bus': 0.25})}

g = Geotype(distbins=distbins) + m

g.appendDemandData(od.ODindex(m, m, 0),
                   od.DemandUnit(distbins[0], demandbydistbin[0], od.Allocation({m: 1.0}), modesplitbydistbin[0]))

g.appendDemandData(od.ODindex(m, m, 1),
                   od.DemandUnit(distbins[1], demandbydistbin[1], od.Allocation({m: 1.0}), modesplitbydistbin[1]))


parking_costs = np.arange(0., 8, 0.1)
road_speeds = np.zeros((np.size(parking_costs)))
car_mode_share = np.zeros((np.size(parking_costs)))

fig, (ax1, ax2) = plt.subplots(1, 2)

for ii in range(np.size(parking_costs)):
    costs = {'car': Costs(0.0003778, parking_costs[ii], 1.0, 1.0), 'bus': Costs(0., 2.5, 0., 1.)}
    m = Microtype(network_params, modeCharacteristics, costs)
    g = Geotype(distbins=distbins) + m
    g.appendDemandData(od.ODindex(m, m, 0),
                       od.DemandUnit(distbins[0], demandbydistbin[0], od.Allocation({m: 1.0}), modesplitbydistbin[0]))
    g.appendDemandData(od.ODindex(m, m, 1),
                       od.DemandUnit(distbins[1], demandbydistbin[1], od.Allocation({m: 1.0}), modesplitbydistbin[1]))
    g.equilibriumModeChoice(20)
    road_speeds[ii] = m.getBaseSpeed()
    car_mode_share[ii] = g.getModeSplit('car')


ax1.plot(parking_costs, car_mode_share)
p_low = ax2.plot(parking_costs, road_speeds)


bus_params = BusParams(road_network_fraction=500, relative_length=3.0,
                                  fixed_density=130. / 100., min_stop_time=15., stop_spacing=1. / 250.,
                                  passenger_wait=5.)

m._mode_characteristics['bus'] = ModeCharacteristics('bus', bus_params)


for ii in range(np.size(parking_costs)):
    costs = {'car': Costs(0.0003778, parking_costs[ii], 1.0, 1.0), 'bus': Costs(0., 2.5, 0., 1.)}
    #m = Microtype(network_params, modeCharacteristics, costs)
    #g = Geotype(distbins=distbins) + m
    m.costs = costs
    g.appendDemandData(od.ODindex(m, m, 0),
                       od.DemandUnit(distbins[0], demandbydistbin[0], od.Allocation({m: 1.0}), modesplitbydistbin[0]))
    g.appendDemandData(od.ODindex(m, m, 1),
                       od.DemandUnit(distbins[1], demandbydistbin[1], od.Allocation({m: 1.0}), modesplitbydistbin[1]))
    g.equilibriumModeChoice(20)
    road_speeds[ii] = m.getBaseSpeed()
    car_mode_share[ii] = g.getModeSplit('car')

ax1.plot(parking_costs, car_mode_share)
ax1.set_xlabel('Parking Cost ($)')
ax1.set_ylabel('Car Mode Share')

p_high = ax2.plot(parking_costs, road_speeds)
ax2.set_xlabel('Parking Cost ($)')
ax2.set_ylabel('Road Speed (m/s)')

plt.legend((p_low[0], p_high[0]), ('Low Bus Service', 'High Bus Service'))