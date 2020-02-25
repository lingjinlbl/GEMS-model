import numpy as np
from utils.network import Network, NetworkCollection, NetworkFlowParams, ModeParams, BusModeParams

network_params_mixed = NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50)
network_params_car = NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50)
network_params_bus = NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50)
network_car = Network(250, network_params_car)
network_car.addMode(ModeParams("car"))
network_bus = Network(750, network_params_bus)
network_bus.addMode(BusModeParams())
network_mixed = Network(500, network_params_mixed)
network_mixed.addMode(BusModeParams())
network_mixed.addMode(ModeParams("car"))

nc = NetworkCollection([network_mixed, network_car, network_bus])
nc.updateMFD()
nc.addVehicles('bus', 1.0)
nc.addVehicles('car', 2.0)
carnc = nc['car']
busnc = nc['bus']
nc.updateMFD()
print('DONE')