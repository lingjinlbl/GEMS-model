import pandas as pd
import numpy as np


def toTransitionMatrix(nhtsTrip, odToPath):
    return None


def interpolateTrips(df, step):
    df = df[~df.index.duplicated()]
    df = df.sort_index()
    df.set_index(df.thru_length_tract.cumsum(), drop=True, inplace=True)
    df = df[~df.index.duplicated()]
    # df['tripEnd'] = True
    # df['tripEnd'].iloc[:-1] = df.trip_id.iloc[1:].values != df.trip_id.iloc[:-1].values
    # finalindex = np.sort(np.hstack([np.arange(0, max(df.index), step), df.index[df.tripEnd].values]))
    finalindex = np.arange(0, max(df.index), step)
    newindex = np.unique(np.hstack([df.index.values, finalindex]))
    out = df.reindex(newindex[newindex >= 0]).interpolate(method='bfill')
    out = out.loc[finalindex]
    out['nextMicrotypeID'] = "exit"
    out['nextTripId'] = -1
    out.iloc[:-1, out.columns.get_loc('nextMicrotypeID')] = out.iloc[1:, :]['MicrotypeID'].values
    out.iloc[:-1, out.columns.get_loc('nextTripId')] = out.iloc[1:, :]['trip_id'].values
    out.loc[out.nextTripId != out.trip_id, 'nextMicrotypeID'] = "exit"
    counts = pd.pivot_table(out, index='MicrotypeID', columns='nextMicrotypeID', values='wtperfin',
                            aggfunc=np.sum).fillna(0)
    counts /= counts.to_numpy().sum(axis=1, keepdims=True)
    return counts.drop(columns=['exit'])


def toDistanceBin(distanceInMiles):
    if distanceInMiles <= 1.3:
        return 'short'
    elif distanceInMiles <= 3:
        return 'medium'
    elif distanceInMiles <= 8:
        return 'long'
    else:
        return 'xlong'


def readAndLabelData(tractToMicrotype, nhtsTrips):
    nhtsTrips = nhtsTrips.merge(tractToMicrotype['MicrotypeID'], left_on='GEOID', right_index=True)
    nhtsTrips = nhtsTrips.merge(tractToMicrotype['MicrotypeID'], left_on='o_geoid', right_index=True,
                                suffixes=('', '_tripOrigin'))
    nhtsTrips = nhtsTrips.merge(tractToMicrotype['MicrotypeID'], left_on='d_geoid', right_index=True,
                                suffixes=('', '_tripDestination'))
    out = nhtsTrips.groupby(
        ['o_geoid', 'd_geoid']).agg({'wtperfin': sum, 'trpmiles': 'mean', 'MicrotypeID_tripOrigin': 'first',
                                     'MicrotypeID_tripDestination': 'first'})
    out['tripDistanceBin'] = out['trpmiles'].fillna(0).apply(toDistanceBin)
    return out


def labelOdToPath(tractToMicrotype, odToPath, nhtsTrips):
    odToPath = pd.merge(odToPath.fillna(-1), nhtsTrips.fillna(-1), left_on=['o_geoid', 'd_geoid'],
                        right_on=['o_geoid', 'd_geoid'])
    odToPath = odToPath.loc[odToPath.thru_length_tract > 0]
    out = pd.merge(odToPath, tractToMicrotype['MicrotypeID'], right_index=True,
                   left_on='thru_geoid', how='inner').set_index(['o_geoid', 'd_geoid', 'order'],
                                                                drop=True).sort_index()[
        ['thru_length_tract', 'MicrotypeID', 'trip_id', 'tripDistanceBin', 'MicrotypeID_tripOrigin',
         'MicrotypeID_tripDestination', 'wtperfin']].dropna()
    return out


if __name__ == "__main__":
    stepSize = 800
    tractToMicrotype = pd.read_csv('data/ccst_geoid_key_transp_geo_with_imputation.csv',
                                   dtype={'GEOID': pd.Int64Dtype()}).set_index('GEOID')
    microtypeIDs = np.sort(tractToMicrotype.MicrotypeID.unique())
    microtypes = pd.DataFrame(microtypeIDs)
    microtypes["avg_through_length"] = stepSize / 1609.34
    microtypes.rename(columns={0: 'MicrotypeID'}).to_csv('output/AvgTripLengths-{0}m.csv'.format(stepSize), index=False)

    nhtsTrips = pd.read_csv('data/nhts_od_pairs_2017_with_ccst_transp_geo.csv', index_col='trip_indx',
                            dtype={'o_geoid': pd.Int64Dtype(), 'd_geoid': pd.Int64Dtype()})
    nhtsTrips = readAndLabelData(tractToMicrotype, nhtsTrips)
    odToPath = pd.read_csv('data/nhts_thru_lengths_ordered_ccst_transp_geo_with_imputation.csv',
                           dtype={'o_geoid': pd.Int64Dtype(), 'd_geoid': pd.Int64Dtype(),
                                  'thru_geoid': pd.Int64Dtype()}).dropna()
    odToPath = labelOdToPath(tractToMicrotype, odToPath, nhtsTrips)
    mat = odToPath.groupby(
        ['MicrotypeID_tripOrigin', 'MicrotypeID_tripDestination', 'tripDistanceBin']).apply(interpolateTrips, stepSize)
    mat.fillna(0).reset_index().rename(
        columns={"MicrotypeID_tripOrigin": "OriginMicrotypeID", "MicrotypeID_tripDestination": "DestMicrotypeID",
                 "tripDistanceBin": "DistanceBinID", "MicrotypeID": "From"}).to_csv(
        'output/TransitionMatrix-{0}m.csv'.format(stepSize))
    print('done')
