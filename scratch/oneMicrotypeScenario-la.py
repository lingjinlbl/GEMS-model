import os
import shutil

import pandas as pd

inFolder = "input-data-losangeles-raw"
outFolder = "input-data-losangeles"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

newDir = os.path.join(ROOT_DIR, "..", outFolder)
# if os.path.exists(newDir):
#     shutil.rmtree(newDir)
# os.makedirs(newDir)

# for copiedFile in ["DistanceBins", "TripGeneration",
#                    "TripPurposes", "ModeExternalities"]:
#     oldPath = os.path.join(ROOT_DIR, "..", inFolder, copiedFile + ".csv")
#     newPath = os.path.join(ROOT_DIR, "..", outFolder, copiedFile + ".csv")
#     shutil.copyfile(oldPath, newPath)

# modesDir = os.path.join(ROOT_DIR, "..", outFolder, "modes")
# if os.path.exists(modesDir):
#     shutil.rmtree(modesDir)
# os.makedirs(modesDir)

# for mode in ["auto", "bike", "bus", "rail", "walk"]:
#     oldPath = os.path.join(ROOT_DIR, "..", inFolder, "modes", mode + ".csv")
#     newPath = os.path.join(ROOT_DIR, "..", outFolder, "modes", mode + ".csv")
#     df = pd.read_csv(oldPath)
#     newdf = df.loc[df.MicrotypeID.str.startswith(geotype), :]
#     newdf.loc[:, "MicrotypeID"] = newdf.loc[:, "MicrotypeID"].str.split('_').str[1].values
#     newdf.sort_values(newdf.columns[0], ascending=True).to_csv(newPath, index=False)
geotype = 'A'
# %% PopulationGroups
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "PopulationGroups.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "PopulationGroups.csv")
df = pd.read_csv(oldPath)
df['BetaTravelTimeMixed'] = 0.0
df.loc[df['Mode'] == 'bike', 'BetaTravelTimeMixed'] = df.loc[df['Mode'] == 'bike', 'BetaTravelTime'].values / 2.
df.loc[df['Mode'] == 'bike', 'BetaTravelTime'] /= 2.0
df.to_csv(newPath, index=False)

# %% TimePeriods
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "TimePeriods.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "TimePeriods.csv")
df = pd.read_csv(oldPath)
df.sort_values(by='TimePeriodID').to_csv(newPath, index=False)

# %% DistanceDistribution
# oldPath = os.path.join(ROOT_DIR, "..", inFolder, "DistanceDistribution.csv")
# newPath = os.path.join(ROOT_DIR, "..", outFolder, "DistanceDistribution.csv")
# df = pd.read_csv(oldPath)
# newdf = df.loc[df.OriginMicrotypeID.str.startswith(geotype) & df.DestinationMicrotypeID.str.startswith(geotype), :]
# newdf.loc[:, "OriginMicrotypeID"] = newdf.loc[:, "OriginMicrotypeID"].str.split('_').str[1].values
# newdf.loc[:, "DestinationMicrotypeID"] = newdf.loc[:, "DestinationMicrotypeID"].str.split('_').str[1].values
# newdf.sort_values(newdf.columns[0], ascending=True).to_csv(newPath, index=False)

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
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "LaneDedicationCost.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "LaneDedicationCost.csv")
df = pd.read_csv(oldPath)
newdf = df.loc[df.MicrotypeID.str.startswith(geotype), :]
newdf.loc[:, "MicrotypeID"] = newdf.loc[:, "MicrotypeID"].str.split('_').str[1].values
newdf.sort_values(newdf.columns[0], ascending=True).to_csv(newPath, index=False)

# # %% Microtypes
# oldPath = os.path.join(ROOT_DIR, "..", inFolder, "Microtypes.csv")
# newPath = os.path.join(ROOT_DIR, "..", outFolder, "Microtypes.csv")
# df = pd.read_csv(oldPath)
# newdf = df.loc[df.MicrotypeID.str.startswith(geotype), :]
# newdf.loc[:, "MicrotypeID"] = newdf.loc[:, "MicrotypeID"].str.split('_').str[1].values
# newdf.sort_values(newdf.columns[0], ascending=True).to_csv(newPath, index=False)

# %% OriginDestination
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "OriginDestination.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "OriginDestination.csv")
df = pd.read_csv(oldPath)
newdf = df.loc[df.HomeMicrotypeID.str.startswith(geotype) & df.OriginMicrotypeID.str.startswith(
    geotype) & df.DestinationMicrotypeID.str.startswith(geotype), :]
newdf.loc[:, "HomeMicrotypeID"] = newdf.loc[:, "HomeMicrotypeID"].str.split('_').str[1].values
newdf.loc[:, "OriginMicrotypeID"] = newdf.loc[:, "OriginMicrotypeID"].str.split('_').str[1].values
newdf.loc[:, "DestinationMicrotypeID"] = newdf.loc[:, "DestinationMicrotypeID"].str.split('_').str[1].values
newdf.sort_values(newdf.columns[0], ascending=True).to_csv(newPath, index=False)

# %% Population
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "Population.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "Population.csv")
df = pd.read_csv(oldPath).dropna()
newdf = df.loc[df.MicrotypeID.str.startswith(geotype), :]
newdf.loc[:, "MicrotypeID"] = newdf.loc[:, "MicrotypeID"].str.split('_').str[1].values
newdf.sort_values(newdf.columns[0], ascending=True).to_csv(newPath, index=False)

# %% RoadNetworkCosts
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "RoadNetworkCosts.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "RoadNetworkCosts.csv")
df = pd.read_csv(oldPath)
newdf = df.loc[df.MicrotypeID.str.startswith(geotype), :]
newdf.loc[:, "MicrotypeID"] = newdf.loc[:, "MicrotypeID"].str.split('_').str[1].values
newdf.sort_values(newdf.columns[0], ascending=True).to_csv(newPath, index=False)

# %% SubNetworks
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "SubNetworks.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "SubNetworks.csv")
df = pd.read_csv(oldPath)
newdf = df.loc[df.MicrotypeID.str.startswith(geotype), :]
newdf.loc[:, "MicrotypeID"] = newdf.loc[:, "MicrotypeID"].str.split('_').str[1].values
newSubNetworks = set(newdf.SubnetworkID.values)
newdf.sort_values(newdf.columns[0], ascending=True).to_csv(newPath, index=False)

# %% ModeToSubNetwork
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "ModeToSubNetwork.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "ModeToSubNetwork.csv")
df = pd.read_csv(oldPath)
newdf = df.loc[df.SubnetworkID.isin(newSubNetworks), :]
newdf.sort_values(newdf.columns[0], ascending=True).to_csv(newPath, index=False)

# %% RoadNetworkCosts
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "TransitionMatrices.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "TransitionMatrices.csv")
df = pd.read_csv(oldPath).set_index(
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
