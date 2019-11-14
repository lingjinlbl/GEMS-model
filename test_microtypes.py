#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from utils.Network import Network
from utils.microtype import Microtype, BusParams, ModeParams, CollectedModeCharacteristics, ModeCharacteristics

network_params_default = Network(0.068, 15.42, 1.88, 0.145, 0.177, 1000, 50)
bus_params_default = BusParams(mean_trip_distance=1000, road_network_fraction=1000, relative_length=3.0,
                                  fixed_density=150. / 100., min_stop_time=15., stop_spacing=1. / 500.,
                                  passenger_wait=5.)

car_params_default = ModeParams(mean_trip_distance=1000, relative_length=1.0)

modeCharacteristics = CollectedModeCharacteristics()
modeCharacteristics['car'] = ModeCharacteristics('car', car_params_default, demand=70 / (10 * 60))
modeCharacteristics['bus'] = ModeCharacteristics('bus', bus_params_default, demand=17 / (10 * 60))

m = Microtype(network_params_default, modeCharacteristics)

total_demands = np.arange(0.02, 0.15, 0.002)
mode_splits = np.arange(0.3, 1.0, 0.01)

average_costs = np.zeros((np.size(total_demands), np.size(mode_splits)))
flows = np.zeros((np.size(total_demands), np.size(mode_splits)))
car_speeds = np.zeros((np.size(total_demands), np.size(mode_splits)))

for ii in range(np.size(total_demands)):
    for jj in range(np.size(mode_splits)):
        car_demand = total_demands[ii] * mode_splits[jj]
        bus_demand = total_demands[ii] * (1.0 - mode_splits[jj])
        modeCharacteristics.setModeDemand('car', car_demand, 1000.0)
        modeCharacteristics.setModeDemand('bus', bus_demand, 1000.0)
        m = Microtype(network_params_default, modeCharacteristics)
        m.findEquilibriumDensityAndSpeed()
        flows[ii, jj] = np.sum(m.getFlows())
        car_speeds[ii, jj] = m.getModeSpeed('car')
        average_costs[ii, jj] = np.sum(m.getTotalTimes()) / np.sum(m.getFlows())

fig1 = plt.figure(figsize=(8, 5))

p1 = plt.contourf(mode_splits, total_demands, average_costs)  # , np.arange(0.08, 0.35, 0.02))
#p2 = plt.contour(bus_demands, car_demands, flows, np.arange(45, 200, 10), cmap='Greys')
cb1 = plt.colorbar(p1)
#cb2 = plt.colorbar(p2)

cb1.set_label('Social Cost per Passenger Trip')
#cb2.set_label('Total Passenger Flow')

plt.xlabel('Bus Demand (trips / time)')
plt.ylabel('Car Demand (trips / time)')

g1 = np.gradient(average_costs)

p3 = plt.contour(mode_splits, total_demands, g1[0] / g1[1], 0, linestyles='dashed', linewidths=2,
                 cmap='Reds')




totalDemand = 0.18
portions = np.arange(0.65, 0.8, 0.005)
oneDemandCosts = np.zeros(np.shape(portions))


bus_params = BusParams(mean_trip_distance=1000, road_network_fraction=500, relative_length=3.0,
                                  fixed_density=100. / 100., min_stop_time=15., stop_spacing=1. / 250.,
                                  passenger_wait=5.)

modeCharacteristics = CollectedModeCharacteristics()
modeCharacteristics['car'] = ModeCharacteristics('car', car_params_default, demand=70 / (10 * 60))
modeCharacteristics['bus'] = ModeCharacteristics('bus', bus_params, demand=17 / (10 * 60))

for ii in range(np.size(portions)):

    modeCharacteristics.setModeDemand('car',  portions[ii] * totalDemand)
    modeCharacteristics.setModeDemand('bus', (1. - portions[ii]) * totalDemand)
    m = Microtype(network_params_default, modeCharacteristics)
    m.findEquilibriumDensityAndSpeed()
    oneDemandCosts[ii] = np.sum(m.getTotalTimes()) / np.sum(m.getFlows())

totalDemand = 0.18
portions = np.arange(0.65, 0.8, 0.005)
oneDemandCosts2 = np.zeros(np.shape(portions))

bus_params = BusParams(mean_trip_distance=1000, road_network_fraction=500, relative_length=3.0,
                                  fixed_density=85. / 100., min_stop_time=15., stop_spacing=1. / 250.,
                                  passenger_wait=5.)
modeCharacteristics = CollectedModeCharacteristics()
modeCharacteristics['car'] = ModeCharacteristics('car', car_params_default, demand=70 / (10 * 60))
modeCharacteristics['bus'] = ModeCharacteristics('bus', bus_params, demand=17 / (10 * 60))

for ii in range(np.size(portions)):
    modeCharacteristics.setModeDemand('car',  portions[ii] * totalDemand)
    modeCharacteristics.setModeDemand('bus', (1. - portions[ii]) * totalDemand)
    m = Microtype(network_params_default, modeCharacteristics)
    m.findEquilibriumDensityAndSpeed()
    oneDemandCosts2[ii] = np.sum(m.getTotalTimes()) / np.sum(m.getFlows())

fig2 = plt.figure(figsize=(7, 4))
plt.plot(portions, oneDemandCosts, label = "More buses")
plt.plot(portions, oneDemandCosts2, label = "Fewer buses")

plt.legend()
plt.xlabel('Portion of trips by car')
plt.ylabel('System average cost')