import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MAKEPLOTS = True
FILENAME = 'combined360.csv'


def readfile(numreps):
    features = pd.read_csv(FILENAME,
                           usecols=['eventid', 'gyro_z_25%', 'gyro_z_50%',
                                    'gyro_z_75%', 'gyro_z_std', 'headway_25%',
                                    'lanepos_25%', 'lanepos_50%',
                                    'lanepos_75%', 'lanepos_std', 'ord',
                                    'speed_25%', 'speed_50%', 'speed_75%',
                                    'speed_std', 'time_bin', 'type', 'state',
                                    'duration', 'laneslope', 'numdeparts'],
                           error_bad_lines=False)
    features['eventid2'] = features['eventid']
    print(features.shape)
    features.loc[features['headway_25%'] < 20, 'eventid'] = np.nan
    features.drop(['headway_25%'], axis=1, inplace=True)
    features.dropna(axis=0, how='any', inplace=True)
    print(features.shape)
    features.loc[features['speed_50%'] < 20, 'eventid'] = np.nan
    features.dropna(axis=0, how='any', inplace=True)
    print(features.shape)
    features.dropna(axis=0, how='any', inplace=True)
    print(features.shape)
    features.reset_index(drop=True, inplace=True)

    grouped = features.groupby('eventid')

    counts = grouped.apply(lambda x: len(x))
    plt.figure(1)
    counts.plot.hist()
    plt.show()

    ordrating = grouped.apply(lambda x: max(x.ord))
    plt.figure(2)
    ordrating.plot.hist()
    plt.show()

    df_features = grouped.apply(input_features_v2, numreps)
#    eventid = pd.Series(df_features.index.levels[0], index=df_features.index);
#    df_features['eventid'] = eventid
    df_features.reset_index(inplace=True, drop=True)

    # export dataframe to csv file
    print("saving features")
    # outfile = os.path.join('C:\\NADS\\github\\DROW',
    #                        'features' + str(numreps) + '.csv')
    outfile = os.path.join(os.getenv('SHRP2ProcessedII'), '..',
                           'features' + str(numreps) + '.csv')
    df_features.to_csv(outfile, index=None)

    return df_features


def input_features(group, numreps):
    # return nothing if too few rows in the group
    if len(group) < numreps:
        return

    # indices to pull features from
    # idx = np.linspace(0, len(group)-1, num=numreps).astype(int)
    idx = np.linspace(len(group)-numreps, len(group)-1,
                      num=numreps).astype(int)

    # construct column names
    varnames = ['gyro_z_50%', 'lanepos_50%', 'laneslope', 'speed_50%']
    columns = [v+'_'+str(i) for v in varnames for (i, ix) in enumerate(idx)]
    columns.append('laneslope_75%')
    columns.append('gyro_z_25%_25%')
    columns.append('gyro_z_75%_75%')
    columns.append('lanepos_25%_25%')
    columns.append('lanepos_75%_75%')
    columns.append('time_bin')
    columns.append('type')
    columns.append('ord')
    columns.append('state')
    columns.append('duration')
    columns.append('numdeparts')

    # take magnitude of the laneslope
    group.loc[:, 'laneslope'] = abs(group['laneslope'])

    # summary stats of group
    df = group.describe()

    # add additional features to the vector
    group.reset_index(drop=True, inplace=True)
    feature_array = group.loc[idx, varnames].values.T.ravel()
    features = np.append(feature_array,
                         [df['laneslope']['75%'], df['gyro_z_25%']['25%'],
                          df['gyro_z_75%']['75%'], df['lanepos_25%']['25%'],
                          df['lanepos_75%']['75%'], max(group['time_bin']),
                          group['type'][0], max(group['ord']),
                          max(group['state']), max(group['duration']),
                          max(group['numdeparts'])])

    df_features = pd.DataFrame(features).T
    df_features.columns = columns

    return df_features


def input_features_v2(group, numreps):
    '''
    this version does away with the time series nature of the feature vector,
    favoring only aggregate measures included in the vector
    '''

    # return nothing if too few rows in the group
    if len(group) < numreps:
        return

    # take magnitude of the laneslope
    group.loc[:, 'laneslope'] = abs(group['laneslope'])

    # summary stats of group
    df = group.describe()

    # add additional features to the vector
    group.reset_index(drop=True, inplace=True)
    df_features = pd.DataFrame({'eventid': df['eventid2']['max'],
                                'slope25': df['laneslope']['25%'],
                                'slope75': df['laneslope']['75%'],
                                'gyro2525': df['gyro_z_25%']['25%'],
                                'gyro2575': df['gyro_z_25%']['75%'],
                                'gyro7525': df['gyro_z_75%']['25%'],
                                'gyro7575': df['gyro_z_75%']['75%'],
                                'lanepos2525': df['lanepos_25%']['25%'],
                                'lanepos2575': df['lanepos_25%']['75%'],
                                'lanepos7525': df['lanepos_75%']['25%'],
                                'lanepos7575': df['lanepos_75%']['75%'],
                                'time': df['time_bin']['max'],
                                'type': group['type'][0],
                                'ord': df['ord']['max'],
                                'state': group['state'][0],
                                'duration': max(group['duration']),
                                'numdeparts': df['numdeparts']['max']},
                               index=[0])

    return df_features


if __name__ == '__main__':
    plt.close('all')

    numreps = 10
    df = readfile(numreps)
    # numreps = 20
    # readfile(numreps)
    # numreps = 30
    # readfile(numreps)
    # numreps = 40
    # readfile(numreps)
    # numreps = 50
    # readfile(numreps)

    plt.pause(1)