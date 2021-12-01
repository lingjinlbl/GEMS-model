import pandas as pd
import os
import numpy as np

miles2meters = 1609.34  # TODO: Change back to right number
headwayFactor = 2 / 24.

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(ROOT_DIR, "..", "input-data-transgeo", "SubNetworks-raw.csv")
subNetworksRaw = pd.read_csv(path, index_col="MicrotypeID")
# subNetworksRaw["Geotype"] = subNetworksRaw["MicrotypeID"].str[0]
# subNetworksRaw["Microtype"] = subNetworksRaw["MicrotypeID"].str[2]

path = os.path.join(ROOT_DIR, "..", "input-data-transgeo", "modes", "bus-raw.csv")
busInput = pd.read_csv(path, index_col="MicrotypeID")
busInput['CoveragePortion'] = 0.0
busInput['Headway'] = 3600 / (busInput['VehicleRevenueMilesPerDay'] / busInput['DirectionalRouteMiles']) / headwayFactor
busInput['VehicleCapacity'] = 40.
busInput['VehicleSize'] = 3.
busInput['StopSpacing'] = 400.
busInput['PassengerWait'] = 6.
busInput['PassengerWaitDedicated'] = 3.
busInput['MinStopTime'] = 7.

path = os.path.join(ROOT_DIR, "..", "input-data-transgeo", "modes", "rail-raw.csv")
railInput = pd.read_csv(path, index_col="MicrotypeID")
railInput['CoveragePortion'] = 0.0
railInput['Headway'] = 3600 / (
        railInput['VehicleRevenueMiles'] / railInput['DirectionalRouteMiles']) / headwayFactor * 365
railInput['Headway'].fillna(7200, inplace=True)
railInput['SpeedInMetersPerSecond'] = railInput['VehicleRevenueMiles'] / railInput['VehicleRevenueHours'] * 1600 / 3600

path = os.path.join(ROOT_DIR, "..", "input-data-transgeo", "modes", "bike-raw.csv")
bikeInput = pd.read_csv(path, index_col="MicrotypeID")
bikeInput['SpeedInMetersPerSecond'] = 4.2
bikeInput['VehicleSize'] = 0.2
bikeInput['DedicatedLanePreference'] = 0.7
bikeInput['PerMileCost'] = bikeInput['PerMinuteCost'] / 60. * bikeInput['SpeedInMetersPerSecond'] * 1609.34

path = os.path.join(ROOT_DIR, "..", "input-data-transgeo", "modes", "walk-raw.csv")
walkInput = pd.read_csv(path, index_col="MicrotypeID")
walkInput['SpeedInMetersPerSecond'] = 1.4
walkInput['VehicleSize'] = 0.2

path = os.path.join(ROOT_DIR, "..", "input-data-transgeo", "modes", "auto-raw.csv")
autoInput = pd.read_csv(path, index_col="MicrotypeID")
autoInput['SpeedInMetersPerSecond'] = 1.4
autoInput['VehicleSize'] = 0.2

"""
SubnetworkID,MicrotypeID,ModesAllowed,Dedicated,Length,MFD,Type,capacityFlow,densityMax,smoothingFactor,vMax,waveSpeed,avgLinkLength

"""

defaults = {'1': {'vMax': 17.0, 'densityMax': 0.15, 'capacityFlow': 0.18, 'smoothingFactor': 0.13, 'waveSpeed': 3.78,
                  'MFD': 'loder', 'avgLinkLength': 50, 'Type': 'Road'},
            '2': {'vMax': 17.0, 'densityMax': 0.15, 'capacityFlow': 0.18, 'smoothingFactor': 0.13, 'waveSpeed': 3.78,
                  'MFD': 'loder', 'avgLinkLength': 50, 'Type': 'Road'},
            '3': {'vMax': 28.0, 'densityMax': 0., 'capacityFlow': 0.5, 'smoothingFactor': 0., 'waveSpeed': 0.,
                  'MFD': 'bottleneck', 'avgLinkLength': 50, 'Type': 'Road'},
            '4': {'vMax': 18.8, 'densityMax': 0., 'capacityFlow': 0.38, 'smoothingFactor': 0., 'waveSpeed': 0.,
                  'MFD': 'bottleneck', 'avgLinkLength': 50, 'Type': 'Road'},
            '5': {'vMax': 18.8, 'densityMax': 0., 'capacityFlow': 0.38, 'smoothingFactor': 0., 'waveSpeed': 0.,
                  'MFD': 'bottleneck', 'avgLinkLength': 50, 'Type': 'Road'},
            '6': {'vMax': 18.8, 'densityMax': 0., 'capacityFlow': 0.38, 'smoothingFactor': 0., 'waveSpeed': 0.,
                  'MFD': 'bottleneck', 'avgLinkLength': 50, 'Type': 'Road'},
            'bus': {'vMax': 17.0, 'capacityFlow': 0.4, 'MFD': 'bottleneck', 'avgLinkLength': 50, 'Type': 'Road'},
            'walk': {'vMax': 1.35, 'MFD': 'fixed', 'Type': 'Sidewalk'},
            'bike': {'vMax': 5., 'MFD': 'fixed', 'Type': 'Road'},
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
    coveragePortion = busInfo.DirectionalRouteMiles / microtypeInfo.LengthNetworkLaneMiles
    busInput.loc[id, "CoveragePortion"] = coveragePortion
    AutoBusBike = {'MicrotypeID': id, 'ModesAllowed': 'Auto-Bus-Bike', 'Dedicated': False, 'Length': lengthInMeters}
    AutoBusBike.update(defaults[microtypeId])
    subNetworksOut[subNetworkId] = pd.Series(AutoBusBike)
    modeToSubNetworkOut.append(
        pd.Series({'ModesAllowed': 'Auto-Bus-Bike', 'SubnetworkID': subNetworkId, 'ModeTypeID': 'auto'}))
    modeToSubNetworkOut.append(
        pd.Series({'ModesAllowed': 'Auto-Bus-Bike', 'SubnetworkID': subNetworkId, 'ModeTypeID': 'bus'}))
    modeToSubNetworkOut.append(
        pd.Series({'ModesAllowed': 'Auto-Bus-Bike', 'SubnetworkID': subNetworkId, 'ModeTypeID': 'bike'}))
    subNetworkId += 1

    Walk = {'MicrotypeID': id, 'ModesAllowed': 'Walk', 'Dedicated': True, 'Length': lengthInMeters / 5.0}
    Walk.update(defaults['walk'])
    subNetworksOut[subNetworkId] = pd.Series(Walk)
    modeToSubNetworkOut.append(
        pd.Series({'ModesAllowed': 'Walk', 'SubnetworkID': subNetworkId, 'ModeTypeID': 'walk'}))
    subNetworkId += 1

    Bus = {'MicrotypeID': id, 'ModesAllowed': 'Bus', 'Dedicated': True, 'Length': 0.}
    Bus.update(defaults['bus'])
    subNetworksOut[subNetworkId] = pd.Series(Bus)
    modeToSubNetworkOut.append(
        pd.Series({'ModesAllowed': 'Bus', 'SubnetworkID': subNetworkId, 'ModeTypeID': 'bus'}))
    subNetworkId += 1

    Bike = {'MicrotypeID': id, 'ModesAllowed': 'Bike', 'Dedicated': True, 'Length': 0.}
    Bike.update(defaults['bike'])
    subNetworksOut[subNetworkId] = pd.Series(Bike)
    modeToSubNetworkOut.append(
        pd.Series({'ModesAllowed': 'Bike', 'SubnetworkID': subNetworkId, 'ModeTypeID': 'bike'}))
    subNetworkId += 1

    Rail = {'MicrotypeID': id, 'ModesAllowed': 'Rail', 'Dedicated': True,
            'Length': railInfo.DirectionalRouteMiles * miles2meters}
    Rail.update(defaults['rail'])
    subNetworksOut[subNetworkId] = pd.Series(Rail)
    railInput.loc[id, "CoveragePortion"] = min(
        [(railInfo.DirectionalRouteMiles) / microtypeInfo.LengthNetworkLaneMiles * 10.0, 1.0])
    modeToSubNetworkOut.append(
        pd.Series({'ModesAllowed': 'Rail', 'SubnetworkID': subNetworkId, 'ModeTypeID': 'rail'}))
    subNetworkId += 1

subNetworksOut = pd.DataFrame(subNetworksOut).transpose().rename_axis("SubnetworkID")
modeToSubNetworkOut = pd.DataFrame(modeToSubNetworkOut)

subNetworksOut.to_csv(os.path.join(ROOT_DIR, "..", "input-data-transgeo", "SubNetworks.csv"))
modeToSubNetworkOut.to_csv(os.path.join(ROOT_DIR, "..", "input-data-transgeo", "ModeToSubNetwork.csv"), index=False)
busInput.to_csv(os.path.join(ROOT_DIR, "..", "input-data-transgeo", "modes", "bus.csv"))
walkInput.to_csv(os.path.join(ROOT_DIR, "..", "input-data-transgeo", "modes", "walk.csv"))
railInput.to_csv(os.path.join(ROOT_DIR, "..", "input-data-transgeo", "modes", "rail.csv"))
bikeInput.to_csv(os.path.join(ROOT_DIR, "..", "input-data-transgeo", "modes", "bike.csv"))
autoInput.to_csv(os.path.join(ROOT_DIR, "..", "input-data-transgeo", "modes", "auto.csv"))

microtypeData = pd.read_csv(os.path.join(ROOT_DIR, "..", "input-data-transgeo", "Microtypes-raw.csv"))
distances = pd.read_csv(os.path.join(ROOT_DIR, "..", "input-data-transgeo", "AvgTripLengths.csv"))

distances.sort_values(by="MicrotypeID").rename(columns={'avg_thru_length': 'DiameterInMiles'}).to_csv(
    os.path.join(ROOT_DIR, "..", "input-data-transgeo", "Microtypes.csv"), index=False)

pd.read_csv(os.path.join(ROOT_DIR, "..", "input-data-transgeo", "TimePeriods.csv")).dropna().sort_values(
    by='TimePeriodID').to_csv(os.path.join(ROOT_DIR, "..", "input-data-transgeo", "TimePeriods.csv"), index=False)

pd.read_csv(os.path.join(ROOT_DIR, "..", "input-data-transgeo", "MicrotypeAssignment.csv")).rename(
    columns={'FromMicrotype': 'FromMicrotypeID', 'ToMicrotype': 'ToMicrotypeID',
             'ThroughMicrotype': 'ThroughMicrotypeID'}).to_csv(
    os.path.join(ROOT_DIR, "..", "input-data-transgeo", "MicrotypeAssignment.csv"), index=False)

print('dpone')
