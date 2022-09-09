import pandas as pd

from model import Model
from utils.optimization import Calibrator, OptimizationVariables

model = Model("input-data-california-A", 1, False)

calibrationVariableNames = [('accessDistanceMultiplier', ('1', 'Bus')),
                            ('accessDistanceMultiplier', ('2', 'Bus')),
                            ('accessDistanceMultiplier', ('3', 'Bus')),
                            ('accessDistanceMultiplier', ('4', 'Bus')),
                            ('accessDistanceMultiplier', ('5', 'Bus')),
                            ('accessDistanceMultiplier', ('6', 'Bus')),
                            ('minStopTime', ('1', 'Bus')),
                            ('minStopTime', ('2', 'Bus')),
                            ('minStopTime', ('3', 'Bus')),
                            ('minStopTime', ('4', 'Bus')),
                            ('minStopTime', ('5', 'Bus')),
                            ('minStopTime', ('6', 'Bus')),
                            ('throughDistanceMultiplier', ('1', '')),
                            ('throughDistanceMultiplier', ('2', '')),
                            ('throughDistanceMultiplier', ('3', '')),
                            ('throughDistanceMultiplier', ('4', '')),
                            ('throughDistanceMultiplier', ('5', '')),
                            ('throughDistanceMultiplier', ('6', '')),
                            ('modeSpeedMPH', ('1', 'Walk')),
                            ('modeSpeedMPH', ('2', 'Walk')),
                            ('modeSpeedMPH', ('3', 'Walk')),
                            ('modeSpeedMPH', ('4', 'Walk')),
                            ('modeSpeedMPH', ('5', 'Walk')),
                            ('modeSpeedMPH', ('6', 'Walk')),
                            ('modeSpeedMPH', ('1', 'Bike')),
                            ('modeSpeedMPH', ('2', 'Bike')),
                            ('modeSpeedMPH', ('3', 'Bike')),
                            ('modeSpeedMPH', ('4', 'Bike')),
                            ('modeSpeedMPH', ('5', 'Bike')),
                            ('modeSpeedMPH', ('6', 'Bike')),
                            ('modeSpeedMPH', ('1', 'Rail')),
                            ('modeSpeedMPH', ('2', 'Rail')),
                            ('modeSpeedMPH', ('3', 'Rail')),
                            ('modeSpeedMPH', ('4', 'Rail')),
                            ('modeSpeedMPH', ('5', 'Rail')),
                            ('modeSpeedMPH', ('6', 'Rail')),
                            ('passengerWait', ('1', 'Bus')),
                            ('passengerWait', ('2', 'Bus')),
                            ('passengerWait', ('3', 'Bus')),
                            ('passengerWait', ('4', 'Bus')),
                            ('passengerWait', ('5', 'Bus')),
                            ('passengerWait', ('6', 'Bus'))]

calibrationVariables = OptimizationVariables(calibrationVariableNames)
calibrator = Calibrator(model, calibrationVariables, regularization=0.2)
result = calibrator.calibrate('trf')
final = calibrationVariables.toPandas(result.x).unstack()
print(final)
calibrator.optimizationVariables.toPandas(result.x).to_csv('calibration-outputs/calibrated-values.csv')
print('done')
