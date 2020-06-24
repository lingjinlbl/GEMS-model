import numpy as np
import matplotlib.pyplot as plt
from utils.microtype import Microtype
from utils.supply import BusParams, ModeParams
from utils.network import Network, NetworkCollection, NetworkFlowParams, Mode, BusMode, BusModeParams


network_params_mixed = NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50)
network_params_car = NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50)
network_params_bus = NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50)
network_car = Network(250, network_params_car)
network_bus = Network(750, network_params_bus)
network_mixed = Network(500, network_params_mixed)

Mode([network_mixed, network_car], ModeParams('car'))
BusMode([network_mixed, network_bus], BusModeParams(0.5))
nc = NetworkCollection([network_mixed, network_car, network_bus])

m = Microtype(nc)
m.setModeDemand('car', 10 / (10 * 60), 1000.0)
m.setModeDemand('bus', 1 / (10 * 60), 1000.0)
print('DONE')
