import os
import shutil

import pandas as pd

geotype = 'B'

inFolder = "input-data-california"
outFolder = "input-data-california-" + geotype
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

newDir = os.path.join(ROOT_DIR, "..", outFolder)
if os.path.exists(newDir):
    shutil.rmtree(newDir)
os.makedirs(newDir)

for copiedFile in ["DistanceBins", "TripGeneration", "TimePeriods",
                   "TripPurposes", "PopulationGroups"]:
    oldPath = os.path.join(ROOT_DIR, "..", inFolder, copiedFile + ".csv")
    newPath = os.path.join(ROOT_DIR, "..", outFolder, copiedFile + ".csv")
    shutil.copyfile(oldPath, newPath)

modesDir = os.path.join(ROOT_DIR, "..", outFolder, "modes")
if os.path.exists(modesDir):
    shutil.rmtree(modesDir)
os.makedirs(modesDir)

shutil.copytree(os.path.join(ROOT_DIR, "..", inFolder, "fleets"), os.path.join(ROOT_DIR, "..", outFolder, "fleets"))

for mode in ["auto", "bike", "bus", "rail", "walk"]:
    oldPath = os.path.join(ROOT_DIR, "..", inFolder, "modes", mode + ".csv")
    newPath = os.path.join(ROOT_DIR, "..", outFolder, "modes", mode + ".csv")
    df = pd.read_csv(oldPath)
    if 'PerEndCost' not in df.columns:
        df['PerEndCost'] = 0.0
    if 'PerMileCost' not in df.columns:
        df['PerMileCost'] = 0.0
    if 'PerStartCost' not in df.columns:
        df['PerStartCost'] = 0.0
    newdf = df.loc[df.MicrotypeID.str.startswith(geotype), :]
    newdf.loc[:, "MicrotypeID"] = newdf.loc[:, "MicrotypeID"].str.split('_').str[1].values
    newdf.sort_values(newdf.columns[0], ascending=True).to_csv(newPath, index=False)

# %% DistanceDistribution
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "DistanceDistribution.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "DistanceDistribution.csv")
df = pd.read_csv(oldPath)
newdf = df.loc[df.OriginMicrotypeID.str.startswith(geotype) & df.DestinationMicrotypeID.str.startswith(geotype), :]
newdf.loc[:, "OriginMicrotypeID"] = newdf.loc[:, "OriginMicrotypeID"].str.split('_').str[1].values
newdf.loc[:, "DestinationMicrotypeID"] = newdf.loc[:, "DestinationMicrotypeID"].str.split('_').str[1].values
newdf.sort_values(newdf.columns[0], ascending=True).to_csv(newPath, index=False)

# %% ModeAvailability
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "ModeAvailability.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "ModeAvailability.csv")
df = pd.read_csv(oldPath)
newdf = df.loc[df.OriginMicrotypeID.str.startswith(geotype) & df.DestinationMicrotypeID.str.startswith(geotype), :]
newdf.loc[:, "OriginMicrotypeID"] = newdf.loc[:, "OriginMicrotypeID"].str.split('_').str[1].values
newdf.loc[:, "DestinationMicrotypeID"] = newdf.loc[:, "DestinationMicrotypeID"].str.split('_').str[1].values
newdf.sort_values(newdf.columns[0], ascending=True).to_csv(newPath, index=False)

# %% MicrotypeAssignment
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "MicrotypeAssignment.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "MicrotypeAssignment.csv")
df = pd.read_csv(oldPath)
newdf = df.loc[df.FromMicrotypeID.str.startswith(geotype) & df.ToMicrotypeID.str.startswith(
    geotype) & df.ThroughMicrotypeID.str.startswith(geotype), :]
newdf.loc[:, "FromMicrotypeID"] = newdf.loc[:, "FromMicrotypeID"].str.split('_').str[1].values
newdf.loc[:, "ToMicrotypeID"] = newdf.loc[:, "ToMicrotypeID"].str.split('_').str[1].values
newdf.loc[:, "ThroughMicrotypeID"] = newdf.loc[:, "ThroughMicrotypeID"].str.split('_').str[1].values
newdf.sort_values(newdf.columns[0], ascending=True).to_csv(newPath, index=False)

# %% LaneDedicationCost
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "RoadNetworkCosts.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "LaneDedicationCost.csv")
df = pd.read_csv(oldPath)
newdf = df.loc[df.MicrotypeID.str.startswith(geotype), :].replace("hv", "auto")
newdf['CostPerMeter'] = newdf['LaneDedicationPerLaneMile'] / 1609.34 / (20 * 365)
newdf.loc[:, "MicrotypeID"] = newdf.loc[:, "MicrotypeID"].str.split('_').str[1].values
newdf.sort_values(newdf.columns[0], ascending=True).to_csv(newPath, index=False)

# %% Microtypes
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "Microtypes.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "Microtypes.csv")
df = pd.read_csv(oldPath)
newdf = df.loc[df.MicrotypeID.str.startswith(geotype), :]
newdf.loc[:, "MicrotypeID"] = newdf.loc[:, "MicrotypeID"].str.split('_').str[1].values
newdf.sort_values(newdf.columns[0], ascending=True).to_csv(newPath, index=False)

# %% OriginDestination
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "OriginDestination.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "OriginDestination.csv")
df = pd.read_csv(oldPath).rename(columns={"PopulationGroupID": "PopulationGroupTypeID"})
newdf = df.loc[df.HomeMicrotypeID.str.startswith(geotype) & df.OriginMicrotypeID.str.startswith(
    geotype) & df.DestinationMicrotypeID.str.startswith(geotype), :]
newdf.loc[:, "HomeMicrotypeID"] = newdf.loc[:, "HomeMicrotypeID"].str.split('_').str[1].values
newdf.loc[:, "OriginMicrotypeID"] = newdf.loc[:, "OriginMicrotypeID"].str.split('_').str[1].values
newdf.loc[:, "DestinationMicrotypeID"] = newdf.loc[:, "DestinationMicrotypeID"].str.split('_').str[1].values
newdf.sort_values(newdf.columns[0], ascending=True).to_csv(newPath, index=False)

# %% Population
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "Population.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "Population.csv")
df = pd.read_csv(oldPath).dropna().rename(columns={"PopulationGroupID": "PopulationGroupTypeID"})
newdf = df.loc[df.MicrotypeID.str.startswith(geotype), :]
newdf.loc[:, "MicrotypeID"] = newdf.loc[:, "MicrotypeID"].str.split('_').str[1].values
newdf.sort_values(newdf.columns[0], ascending=True).to_csv(newPath, index=False)

# %% RoadNetworkCosts
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "RoadNetworkCosts.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "RoadNetworkCosts.csv")
df = pd.read_csv(oldPath)
newdf = df.loc[df.MicrotypeID.str.startswith(geotype), :].replace("hv", "auto")
newdf.loc[:, "MicrotypeID"] = newdf.loc[:, "MicrotypeID"].str.split('_').str[1].values
newdf.sort_values(newdf.columns[0], ascending=True).to_csv(newPath, index=False)

# %% FreightDemand
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "FreightDemand.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "FreightDemand.csv")
df = pd.read_csv(oldPath)
newdf = df.loc[df.MicrotypeID.str.startswith(geotype), :]
newdf.loc[:, "MicrotypeID"] = newdf.loc[:, "MicrotypeID"].str.split('_').str[1].values
newdf.sort_values(newdf.columns[0], ascending=True).to_csv(newPath, index=False)

# %% SubNetworks
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "SubNetworks.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "SubNetworks.csv")
df = pd.read_csv(oldPath)
newdf = df.loc[df.MicrotypeID.str.startswith(geotype), :].replace("hv", "auto")
newdf.loc[:, "MicrotypeID"] = newdf.loc[:, "MicrotypeID"].str.split('_').str[1].values
newSubNetworks = set(newdf.SubnetworkID.values)
newdf.sort_values(newdf.columns[0], ascending=True).to_csv(newPath, index=False)

# %% ModeToSubNetwork
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "ModeToSubNetwork.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "ModeToSubNetwork.csv")
df = pd.read_csv(oldPath)
newdf = df.loc[df.SubnetworkID.isin(newSubNetworks), :].replace("hv", "auto")
newdf.sort_values(newdf.columns[0], ascending=True).to_csv(newPath, index=False)

# %% ModeExternalities
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "Externalities.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "ModeExternalities.csv")
df = pd.read_csv(oldPath)
newdf = df.loc[df.MicrotypeID.str.startswith(geotype), :].rename(
    columns={'PerMileExtCost': 'CostPerVehicleMile'}).replace("hv", "auto").replace("freight", "freight_combo")
newdf['CostPerPassengerMile'] = 0.0
newdf.loc[:, "MicrotypeID"] = newdf.loc[:, "MicrotypeID"].str.split('_').str[1].values
otherdf = newdf.loc[newdf['Mode'] == "freight_combo"].copy().replace("freight_combo", "freight_single")
newdf = pd.concat([newdf, otherdf])
newdf.sort_values(newdf.columns[0], ascending=True).to_csv(newPath, index=False)

# %% RoadNetworkCosts
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "TransitionMatrix.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "TransitionMatrices.csv")
df = pd.read_csv(oldPath).rename(columns={"DestMicrotypeID": "DestinationMicrotypeID"}).set_index(
    ["OriginMicrotypeID", "DestinationMicrotypeID", "DistanceBinID", "From"])
newdf = df.loc[df.index.get_level_values(0).str.startswith(geotype) & df.index.get_level_values(1).str.startswith(
    geotype) & df.index.get_level_values(3).str.startswith(geotype), df.columns.str.startswith(geotype)]
newdf.reset_index(inplace=True)
newdf.loc[:, "OriginMicrotypeID"] = newdf.loc[:, "OriginMicrotypeID"].str.split('_').str[1].values
newdf.loc[:, "DestinationMicrotypeID"] = newdf.loc[:, "DestinationMicrotypeID"].str.split('_').str[1].values
newdf.loc[:, "From"] = newdf.loc[:, "From"].str.split('_').str[1].values
newdf.columns = newdf.columns.str.split('_').str[-1].values
newdf.sort_values(newdf.columns[0], ascending=True).to_csv(newPath, index=False)

print("AA")
