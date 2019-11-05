#!/usr/bin/env python3
import utils.IO as io
import numpy as np
import matplotlib.pyplot as plt

from utils.microtype import Microtype

network_params_default = io.Network(0.068, 15.42, 1.88, 0.145, 0.177, 1000, 50)
bus_params_default = io.BusParams(mean_trip_distance=1000, road_network_fraction=500, relative_length=3.0,
                                  fixed_density=95. / 100., min_stop_time=15., stop_spacing=1. / 250.,
                                  passenger_wait=5.)

car_params_default = io.ModeParams(mean_trip_distance=1000, relative_length=1.0)

modeCharacteristics = io.CollectedModeCharacteristics()
modeCharacteristics['car'] = io.ModeCharacteristics('car', car_params_default, demand=70 / (10 * 60))
modeCharacteristics['bus'] = io.ModeCharacteristics('bus', bus_params_default, demand=17 / (10 * 60))

m = Microtype(network_params_default, modeCharacteristics)

car_demands = np.arange(0.05, 0.13, 0.001)
bus_demands = np.arange(0.001, 0.08, 0.001)

average_costs = np.zeros((np.size(car_demands), np.size(bus_demands)))
flows = np.zeros((np.size(car_demands), np.size(bus_demands)))
car_speeds = np.zeros((np.size(car_demands), np.size(bus_demands)))

for ii in range(np.size(car_demands)):
    for jj in range(np.size(bus_demands)):
        modeCharacteristics.setModeDemand('car', car_demands[ii])
        modeCharacteristics.setModeDemand('bus', bus_demands[jj])
        m = Microtype(network_params_default, modeCharacteristics)
        m.findEquilibriumDensityAndSpeed()
        flows[ii, jj] = np.sum(m.getFlows())
        car_speeds[ii, jj] = m.getModeSpeed('car')
        average_costs[ii, jj] = np.sum(m.getTotalTimes()) / np.sum(m.getFlows())

fig1 = plt.figure(figsize=(8, 5))

p1 = plt.contourf(bus_demands, car_demands, average_costs)  # , np.arange(0.08, 0.35, 0.02))
p2 = plt.contour(bus_demands, car_demands, flows, np.arange(45, 200, 10), cmap='Greys')
cb1 = plt.colorbar(p1)
cb2 = plt.colorbar(p2)

cb1.set_label('Social Cost per Passenger Trip')
cb2.set_label('Total Passenger Flow')

plt.xlabel('Bus Demand (trips / time)')
plt.ylabel('Car Demand (trips / time)')

g1 = np.gradient(average_costs)
g2 = np.gradient(flows)

p3 = plt.contour(bus_demands, car_demands, g1[0] / g1[1] - g2[0] / g2[1], 0, linestyles='dashed', linewidths=2,
                 cmap='Reds')




totalDemand = 0.16
portions = np.arange(0.5, 1.0, 0.02)
oneDemandCosts = np.zeros(np.shape(portions))

bus_params = io.BusParams(mean_trip_distance=1000, road_network_fraction=500, relative_length=3.0,
                                  fixed_density=95. / 100., min_stop_time=15., stop_spacing=1. / 250.,
                                  passenger_wait=5.)
modeCharacteristics['bus'] = io.ModeCharacteristics('bus', bus_params, demand=17 / (10 * 60))

for ii in range(np.size(portions)):

    modeCharacteristics.setModeDemand('car',  portions[ii] * totalDemand)
    modeCharacteristics.setModeDemand('bus', (1. - portions[ii]) * totalDemand)
    m = Microtype(network_params_default, modeCharacteristics)
    m.findEquilibriumDensityAndSpeed()
    oneDemandCosts[ii] = np.sum(m.getTotalTimes()) / np.sum(m.getFlows())

totalDemand = 0.16
portions = np.arange(0.5, 1.0, 0.02)
oneDemandCosts2 = np.zeros(np.shape(portions))

bus_params = io.BusParams(mean_trip_distance=1000, road_network_fraction=500, relative_length=3.0,
                                  fixed_density=85. / 100., min_stop_time=15., stop_spacing=1. / 250.,
                                  passenger_wait=5.)
modeCharacteristics['bus'] = io.ModeCharacteristics('bus', bus_params, demand=17 / (10 * 60))

fig2 = plt.figure(figsize=(8, 5))
plt.plot(portions, oneDemandCosts)

for ii in range(np.size(portions)):

    modeCharacteristics.setModeDemand('car',  portions[ii] * totalDemand)
    modeCharacteristics.setModeDemand('bus', (1. - portions[ii]) * totalDemand)
    m = Microtype(network_params_default, modeCharacteristics)
    m.findEquilibriumDensityAndSpeed()
    oneDemandCosts2[ii] = np.sum(m.getTotalTimes()) / np.sum(m.getFlows())

plt.plot(portions, oneDemandCosts2)
