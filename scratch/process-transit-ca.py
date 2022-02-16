import pandas as pd
import os
import numpy as np
import shutil

miles2meters = 1609.34
headwayFactor = 2 / 24.
interliningFactor = 3.0
operatingHoursPerDay = 20

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(ROOT_DIR, "..", "input-data-california-raw", "SubNetworks-raw.csv")
subNetworksRaw = pd.read_csv(path, index_col="MicrotypeID")
# subNetworksRaw["Geotype"] = subNetworksRaw["MicrotypeID"].str[0]
# subNetworksRaw["Microtype"] = subNetworksRaw["MicrotypeID"].str[2]

path = os.path.join(ROOT_DIR, "..", "input-data-california-raw", "modes", "bus-raw.csv")
busInput = pd.read_csv(path, index_col="MicrotypeID")
busInput['coveragePortion'] = 0.0
busInput['Headway'] = 3600 / (busInput['VehicleRevenueMilesPerDay'] / busInput[
    'DirectionalRouteMiles']) / headwayFactor / interliningFactor
busInput['VehicleCapacity'] = 40.
busInput['VehicleSize'] = 3.
busInput['StopSpacing'] = 400.
busInput['PassengerWait'] = 6.
busInput['PassengerWaitDedicated'] = 3.
busInput['MinStopTime'] = 7.
busInput['VehicleOperatingCostPerHour'] = busInput['DailyCostPerVehicle'] / operatingHoursPerDay

path = os.path.join(ROOT_DIR, "..", "input-data-california-raw", "modes", "bike-raw.csv")
bikeInput = pd.read_csv(path, index_col="MicrotypeID")
bikeInput['SpeedInMetersPerSecond'] = 4.2
bikeInput['VehicleSize'] = 0.2
bikeInput['DedicatedLanePreference'] = 0.7
bikeInput['PerMileCost'] = bikeInput['PerMinuteCost'] / 60. * bikeInput['SpeedInMetersPerSecond'] * 1609.34
bikeInput['BikesPerCapita'] = bikeInput['StationDensity'].copy()

path = os.path.join(ROOT_DIR, "..", "input-data-california-raw", "modes", "walk-raw.csv")
walkInput = pd.read_csv(path, index_col="MicrotypeID")
walkInput['SpeedInMetersPerSecond'] = 1.4
walkInput['VehicleSize'] = 0.2

path = os.path.join(ROOT_DIR, "..", "input-data-california-raw", "modes", "auto-raw.csv")
autoInput = pd.read_csv(path, index_col="MicrotypeID")
autoInput['SpeedInMetersPerSecond'] = 1.4
autoInput['VehicleSize'] = 0.2

path = os.path.join(ROOT_DIR, "..", "input-data-california-raw", "modes", "rail-raw.csv")
railInput = pd.read_csv(path, index_col="MicrotypeID")
railInput['CoveragePortion'] = 0.0
railInput['Headway'] = 3600 / (
        railInput['VehicleRevenueMilesPerDay'] / railInput['DirectionalRouteMiles']) / headwayFactor * 365
railInput['Headway'].fillna(7200, inplace=True)
railInput['SpeedInMetersPerSecond'] = railInput['VehicleRevenueMilesPerDay'] / railInput[
    'VehicleRevenueHoursPerDay'] * 1600 / 3600
railInput['StopSpacing'] = 1200.
railInput['VehicleOperatingCostPerHour'] = railInput['DailyCostPerVehicle'] / operatingHoursPerDay

"""
SubnetworkID,MicrotypeID,ModesAllowed,Dedicated,Length,MFD,Type,capacityFlow,densityMax,smoothingFactor,vMax,waveSpeed,avgLinkLength

"""

defaults = {'1': {'a': -100.0, 'criticalDensity': 0.118, 'densityMax': 0.15,
                  'MFD': 'modified-quadratic', 'avgLinkLength': 50, 'Type': 'Road', 'smoothingFactor': np.nan,
                  'waveSpeed': np.nan},
            '2': {'a': -182.0, 'criticalDensity': 0.085, 'densityMax': 0.15,
                  'MFD': 'modified-quadratic', 'avgLinkLength': 50, 'Type': 'Road'},
            '3': {'vMax': 14.8, 'capacityFlow': 0.344,
                  'MFD': 'bottleneck', 'avgLinkLength': 50, 'Type': 'Road'},
            '4': {'vMax': 14.5, 'capacityFlow': 0.305,
                  'MFD': 'bottleneck', 'avgLinkLength': 50, 'Type': 'Road'},
            '5': {'vMax': 15.5, 'capacityFlow': 0.292,
                  'MFD': 'bottleneck', 'avgLinkLength': 50, 'Type': 'Road'},
            '6': {'vMax': 19.4, 'capacityFlow': 0.325,
                  'MFD': 'bottleneck', 'avgLinkLength': 50, 'Type': 'Road'},
            'bus': {'vMax': 17.0, 'capacityFlow': 0.4, 'MFD': 'bottleneck', 'avgLinkLength': 50, 'Type': 'Road'},
            'walk': {'vMax': 1.35, 'MFD': 'fixed', 'Type': 'Sidewalk'},
            'bike': {'vMax': 5., 'densityMax': 0.15, 'MFD': 'fixed', 'Type': 'Road'},
            'rail': {'vMax': 20., 'MFD': 'fixed', 'Type': 'Rail'}
            }

subNetworksOut = dict()
modeToSubNetworkOut = []
subNetworkId = 0

for id, microtypeInfo in subNetworksRaw.iterrows():
    microtypeId = id[-1]
    busInfo = busInput.loc[id]
    railInfo = railInput.loc[id]
    lengthInMeters = microtypeInfo.LengthNetworkLaneMiles * miles2meters
    coveragePortion = busInfo.DirectionalRouteMiles / microtypeInfo.LengthNetworkLaneMiles / interliningFactor
    busInput.loc[id, "CoveragePortion"] = coveragePortion
    AutoBusBike = {'MicrotypeID': id, 'ModesAllowed': 'Auto-Bus-Bike', 'Dedicated': False, 'Length': lengthInMeters}
    AutoBusBike.update(defaults[microtypeId])
    subNetworksOut[subNetworkId] = pd.Series(AutoBusBike)
    modeToSubNetworkOut.append(
        pd.Series({'ModesAllowed': 'Auto-Bus-Bike-Freight', 'SubnetworkID': subNetworkId, 'Mode': 'auto'}))
    modeToSubNetworkOut.append(
        pd.Series({'ModesAllowed': 'Auto-Bus-Bike-Freight', 'SubnetworkID': subNetworkId, 'Mode': 'bus'}))
    modeToSubNetworkOut.append(
        pd.Series({'ModesAllowed': 'Auto-Bus-Bike-Freight', 'SubnetworkID': subNetworkId, 'Mode': 'bike'}))
    modeToSubNetworkOut.append(
        pd.Series({'ModesAllowed': 'Auto-Bus-Bike-Freight', 'SubnetworkID': subNetworkId, 'Mode': 'freight_combo'}))
    modeToSubNetworkOut.append(
        pd.Series({'ModesAllowed': 'Auto-Bus-Bike-Freight', 'SubnetworkID': subNetworkId, 'Mode': 'freight_single'}))
    subNetworkId += 1

    Walk = {'MicrotypeID': id, 'ModesAllowed': 'Walk', 'Dedicated': True, 'Length': lengthInMeters / 5.0}
    Walk.update(defaults['walk'])
    subNetworksOut[subNetworkId] = pd.Series(Walk)
    modeToSubNetworkOut.append(
        pd.Series({'ModesAllowed': 'Walk', 'SubnetworkID': subNetworkId, 'Mode': 'walk'}))
    subNetworkId += 1

    Bus = {'MicrotypeID': id, 'ModesAllowed': 'Bus', 'Dedicated': True, 'Length': 0.}
    Bus.update(defaults['bus'])
    subNetworksOut[subNetworkId] = pd.Series(Bus)
    modeToSubNetworkOut.append(
        pd.Series({'ModesAllowed': 'Bus', 'SubnetworkID': subNetworkId, 'Mode': 'bus'}))
    subNetworkId += 1

    Bike = {'MicrotypeID': id, 'ModesAllowed': 'Bike', 'Dedicated': True, 'Length': 0.}
    Bike.update(defaults['bike'])
    subNetworksOut[subNetworkId] = pd.Series(Bike)
    modeToSubNetworkOut.append(
        pd.Series({'ModesAllowed': 'Bike', 'SubnetworkID': subNetworkId, 'Mode': 'bike'}))
    subNetworkId += 1

    Rail = {'MicrotypeID': id, 'ModesAllowed': 'Rail', 'Dedicated': True,
            'Length': railInfo.DirectionalRouteMiles * miles2meters}
    Rail.update(defaults['rail'])
    subNetworksOut[subNetworkId] = pd.Series(Rail)
    railInput.loc[id, "CoveragePortion"] = min(
        [railInfo.DirectionalRouteMiles / microtypeInfo.LengthNetworkLaneMiles * 10.0, 1.0])
    modeToSubNetworkOut.append(
        pd.Series({'ModesAllowed': 'Rail', 'SubnetworkID': subNetworkId, 'Mode': 'rail'}))
    subNetworkId += 1

subNetworksOut = pd.DataFrame(subNetworksOut).transpose().rename_axis("SubnetworkID")
modeToSubNetworkOut = pd.DataFrame(modeToSubNetworkOut)

subNetworksOut.to_csv(os.path.join(ROOT_DIR, "..", "input-data-california", "SubNetworks.csv"))
modeToSubNetworkOut.to_csv(os.path.join(ROOT_DIR, "..", "input-data-california", "ModeToSubNetwork.csv"),
                           index=False)
busInput.to_csv(os.path.join(ROOT_DIR, "..", "input-data-california", "modes", "bus.csv"))
walkInput.to_csv(os.path.join(ROOT_DIR, "..", "input-data-california", "modes", "walk.csv"))
bikeInput.to_csv(os.path.join(ROOT_DIR, "..", "input-data-california", "modes", "bike.csv"))
autoInput.to_csv(os.path.join(ROOT_DIR, "..", "input-data-california", "modes", "auto.csv"))
railInput.to_csv(os.path.join(ROOT_DIR, "..", "input-data-california", "modes", "rail.csv"))

microtypeData = pd.read_csv(os.path.join(ROOT_DIR, "..", "input-data-california-raw", "Microtypes.csv"))
distances = pd.read_csv(os.path.join(ROOT_DIR, "..", "input-data-california-raw", "AvgTripLengths.csv"))

distances.sort_values(by="MicrotypeID").rename(columns={'avg_thru_length': 'DiameterInMiles'}).to_csv(
    os.path.join(ROOT_DIR, "..", "input-data-california", "Microtypes.csv"), index=False)

pd.read_csv(os.path.join(ROOT_DIR, "..", "input-data-california-raw", "TimePeriods.csv")).dropna().sort_values(
    by='TimePeriodID').to_csv(os.path.join(ROOT_DIR, "..", "input-data-california", "TimePeriods.csv"), index=False)

pd.read_csv(os.path.join(ROOT_DIR, "..", "input-data-california-raw", "MicrotypeAssignment.csv")).rename(
    columns={'FromMicrotype': 'FromMicrotypeID', 'ToMicrotype': 'ToMicrotypeID',
             'ThroughMicrotype': 'ThroughMicrotypeID'}).to_csv(
    os.path.join(ROOT_DIR, "..", "input-data-california", "MicrotypeAssignment.csv"), index=False)

# %% PopulationGroups
oldPath = os.path.join(ROOT_DIR, "..", "input-data-california-raw", "PopulationGroups.csv")
newPath = os.path.join(ROOT_DIR, "..", "input-data-california", "PopulationGroups.csv")
df = pd.read_csv(oldPath).rename(columns={"PopulationGroupID": "PopulationGroupTypeID"}).replace({'hv': 'auto'})
df['Mode'] = df['Mode'].str.lower()
df['BetaTravelTimeMixed'] = 0.0
df.loc[df['Mode'] == 'bike', 'BetaTravelTimeMixed'] = df.loc[df['Mode'] == 'bike', 'BetaTravelTime'].values / 2.
df.loc[df['Mode'] == 'bike', 'BetaTravelTime'] /= 2.0
df['BetaWaitTime'] = df['BetaWaitAccessTime']
df['BetaAccessTime'] = df['BetaWaitAccessTime']
df['BetaWaitTime_Pooled'] = df['BetaWaitAccessTime_Pooled']
df['BetaAccessTime_Pooled'] = df['BetaWaitAccessTime_Pooled']
df['BetaWaitTime_Pooled_EW'] = df['BetaWaitAccessTime_Pooled_EW']
df['BetaAccessTime_Pooled_EW'] = df['BetaWaitAccessTime_Pooled_EW']
df.to_csv(newPath, index=False)

# %% TimePeriods
oldPath = os.path.join(ROOT_DIR, "..", "input-data-california-raw", "TimePeriods.csv")
newPath = os.path.join(ROOT_DIR, "..", "input-data-california", "TimePeriods.csv")
df = pd.read_csv(oldPath)
df.sort_values(by='TimePeriodID').to_csv(newPath, index=False)

# %% TripGeneration
oldPath = os.path.join(ROOT_DIR, "..", "input-data-california-raw", "TripGeneration.csv")
newPath = os.path.join(ROOT_DIR, "..", "input-data-california", "TripGeneration.csv")
df = pd.read_csv(oldPath).rename(columns={"PopulationGroupID": "PopulationGroupTypeID"})
df.to_csv(newPath, index=False)

newDir = os.path.join(ROOT_DIR, "..", "input-data-california", "fleets")
if os.path.exists(newDir):
    shutil.rmtree(newDir)
os.makedirs(newDir)

freight = pd.DataFrame(
    {'freight_single': {'VehicleSize': 2.0, 'MaximumSpeedMetersPerSecond': 20.0, 'OperatingCostPerHour': 30.0},
     'freight_combo': {'VehicleSize': 4.0, 'MaximumSpeedMetersPerSecond': 20.0,
                       'OperatingCostPerHour': 40.0}}).transpose()
freight.index.name = "Fleet"
freight.to_csv(os.path.join(newDir, 'freight.csv'))
print('done')
