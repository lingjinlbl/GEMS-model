import pandas as pd
import os
import numpy as np

miles2meters = 1609.34  # TODO: Change back to right number
headwayFactor = 2 / 24.

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(ROOT_DIR, "..", "input-data-losangeles-raw", "SubNetworks-raw.csv")
subNetworksRaw = pd.read_csv(path, index_col="MicrotypeID")
# subNetworksRaw["Geotype"] = subNetworksRaw["MicrotypeID"].str[0]
# subNetworksRaw["Microtype"] = subNetworksRaw["MicrotypeID"].str[2]

path = os.path.join(ROOT_DIR, "..", "input-data-transgeo", "modes", "bus-raw.csv")
busInput = pd.read_csv(path, index_col="MicrotypeID")
busInput['coveragePortion'] = 0.0
busInput['Headway'] = 3600 / (busInput['VehicleRevenueMilesPerDay'] / busInput['DirectionalRouteMiles']) / headwayFactor
busInput['VehicleCapacity'] = 40.
busInput['VehicleSize'] = 3.
busInput['StopSpacing'] = 400.
busInput['PassengerWait'] = 6.
busInput['PassengerWaitDedicated'] = 3.
busInput['MinStopTime'] = 7.

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

defaults = {'1': {'a': -100.0, 'criticalDensity': 0.118, 'densityMax': 0.15,
                  'MFD': 'modified-quadratic', 'avgLinkLength': 50, 'Type': 'Road'},
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
            'bike': {'vMax': 5., 'densityMax': 0.15, 'MFD': 'fixed', 'Type': 'Road'}
            }

subNetworksOut = dict()
modeToSubNetworkOut = []
subNetworkId = 0

for id, microtypeInfo in subNetworksRaw.iterrows():
    microtypeId = id[-1]
    busInfo = busInput.loc[id]

    lengthInMeters = microtypeInfo.LengthNetworkLaneMiles * miles2meters
    coveragePortion = busInfo.DirectionalRouteMiles / microtypeInfo.LengthNetworkLaneMiles
    busInput.loc[id, "CoveragePortion"] = coveragePortion
    AutoBusBike = {'MicrotypeID': id, 'ModesAllowed': 'Auto-Bus-Bike', 'Dedicated': False, 'Length': lengthInMeters}
    AutoBusBike.update(defaults[microtypeId])
    subNetworksOut[subNetworkId] = pd.Series(AutoBusBike)
    modeToSubNetworkOut.append(
        pd.Series({'ModesAllowed': 'Auto-Bus-Bike', 'SubnetworkID': subNetworkId, 'Mode': 'auto'}))
    modeToSubNetworkOut.append(
        pd.Series({'ModesAllowed': 'Auto-Bus-Bike', 'SubnetworkID': subNetworkId, 'Mode': 'bus'}))
    modeToSubNetworkOut.append(
        pd.Series({'ModesAllowed': 'Auto-Bus-Bike', 'SubnetworkID': subNetworkId, 'Mode': 'bike'}))
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

subNetworksOut = pd.DataFrame(subNetworksOut).transpose().rename_axis("SubnetworkID")
modeToSubNetworkOut = pd.DataFrame(modeToSubNetworkOut)

subNetworksOut.to_csv(os.path.join(ROOT_DIR, "..", "input-data-losangeles-raw", "SubNetworks.csv"))
modeToSubNetworkOut.to_csv(os.path.join(ROOT_DIR, "..", "input-data-losangeles-raw", "ModeToSubNetwork.csv"),
                           index=False)
busInput.to_csv(os.path.join(ROOT_DIR, "..", "input-data-losangeles-raw", "modes", "bus.csv"))
walkInput.to_csv(os.path.join(ROOT_DIR, "..", "input-data-losangeles-raw", "modes", "walk.csv"))
bikeInput.to_csv(os.path.join(ROOT_DIR, "..", "input-data-losangeles-raw", "modes", "bike.csv"))
autoInput.to_csv(os.path.join(ROOT_DIR, "..", "input-data-losangeles-raw", "modes", "auto.csv"))

microtypeData = pd.read_csv(os.path.join(ROOT_DIR, "..", "input-data-losangeles-raw", "Microtypes-raw.csv"))
distances = pd.read_csv(os.path.join(ROOT_DIR, "..", "input-data-losangeles-raw", "AvgTripLengths.csv"))

distances.sort_values(by="MicrotypeID").rename(columns={'avg_thru_length': 'DiameterInMiles'}).to_csv(
    os.path.join(ROOT_DIR, "..", "input-data-losangeles-raw", "Microtypes.csv"), index=False)

pd.read_csv(os.path.join(ROOT_DIR, "..", "input-data-losangeles-raw", "TimePeriods.csv")).dropna().sort_values(
    by='TimePeriodID').to_csv(os.path.join(ROOT_DIR, "..", "input-data-losangeles-raw", "TimePeriods.csv"), index=False)

pd.read_csv(os.path.join(ROOT_DIR, "..", "input-data-losangeles-raw", "MicrotypeAssignment.csv")).rename(
    columns={'FromMicrotype': 'FromMicrotypeID', 'ToMicrotype': 'ToMicrotypeID',
             'ThroughMicrotype': 'ThroughMicrotypeID'}).to_csv(
    os.path.join(ROOT_DIR, "..", "input-data-losangeles-raw", "MicrotypeAssignment.csv"), index=False)

print('dpone')