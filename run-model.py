import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import Model, Optimizer

model = Model("input-data-losangeles-national-params", nSubBins=1)
optimizer = Optimizer(model, modesAndMicrotypes=[('1', 'Bus'), ('2', 'Bus')],  # These determine headways
                      fromToSubNetworkIDs=[('1', 'Bus'), ('1', 'Bike'), ('2', 'Bus'), ('2', 'Bike')],
                      # These determine ROW allocation
                      method="min")

# // Run the model with default parameters

x0 = optimizer.x0()

# x0 holds the decision variables. In this example, they are
# [Portion bus ROW in Microtype 1, bike Row in 1, Bus ROW in 2, Bike ROW in 2, Bus headway
# in 1 (in seconds / 100), Bus headway in 2)].
# These are defined in the Optimizer definition above

optimizer.evaluate(x0)
modeSplit, speed, utility = model.toPandas()
allCosts = optimizer.sumAllCosts()

# // Update the model to allocate 10% of bus route to bus lanes in microtype 1
optimizer.evaluate([0.1, 0., 0., 0., 0.3, 0.3])
modeSplit2, speed2, utility2 = model.toPandas()
allCosts2 = optimizer.sumAllCosts()

# // Run the full optimizer (this can take a long time!)
optimizer.minimize()
modeSplit_opt, speed_opt, utility_opt = model.toPandas()
allCosts_opt = optimizer.sumAllCosts()
print('done')
