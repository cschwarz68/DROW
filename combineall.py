import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import linregress

FS = 10
FS_TIMESTAMP = 1000
SEGLENGTH = 6 * FS_TIMESTAMP
LANEWIDTH = 365.76
CARWIDTH = 182.88
KPH2CMPS = 27.78
WRITEFILE = True
PATH = os.getenv('SHRP2ProcessedII')


def combinefiles():
    '''
    loop through all the csv files in the SHRP2 processed path,
    get summary statistics from each one
    '''

    # list of csv files
    files = glob.glob(os.path.join(PATH, 'File_ID_*.csv'))

    dflist = []

    # loop through all the csv files and process each one,
    for fullfile in files:
        drive, path = os.path.splitdrive(fullfile)
        path, filename = os.path.split(path)
        filename, file_extension = os.path.splitext(filename)
        print(filename)
        # if filename != 'File_ID_41894439':
        #     continue

        df = pd.read_csv(fullfile,
                         usecols=['timestamp', 'time_bin', 'canspeed',
                                  'acc_x', 'acc_y', 'gyro_z', 'lanepos',
                                  'lanewidth', 'lv_id', 'lv_headway',
                                  'lv_range_rate', 'eventid', 'ord', 'type',
                                  'state', 'duration'],
                         error_bad_lines=False)

        grouped = df.groupby('eventid')
        df_measures = grouped.apply(measures_by_event)

        if len(df_measures) > 0:
            print('df_measures length is ' + str(len(df_measures)))
            dflist.append(df_measures)
            print('dflist length is ' + str(len(dflist)))

    df_combined = pd.concat(dflist, axis=0)
    print('df_combined length is ' + str(len(df_combined)))

    # save combined data to a file
    if WRITEFILE:
        # export dataframe to csv file
        print("saving combined data to combined.csv")
        # outfile = os.path.join('C:\\NADS\\github\\DROW','combined.csv')
        outfile = os.path.join(os.getenv('SHRP2ProcessedII'), '..',
                               'combined.csv')
        df_combined.to_csv(outfile)

    return df_combined


def measures_by_event(group):
    group.reset_index(inplace=True, drop=True)
    group['time'] = group.loc[:, 'timestamp'] - group.loc[0, 'timestamp']
    group['segment'] = group['time'] / SEGLENGTH
    group = group.astype({'segment': int})
    # numsegs = 60 * FS_TIMESTAMP / SEGLENGTH
    # if max(group['segment']) != numsegs-1:
    #     return None
    group['canspeed_cmps'] = group['canspeed'] * KPH2CMPS
    num_nans = len(np.where(np.isnan(group['canspeed']))[0])
    num_tot = len(group)
    if float(num_nans) / float(num_tot) > 0.9:
        print('by event: too many nans, returning prematurely')
        return
    group['dist_cm'] = group['canspeed_cmps'].cumsum() / FS

    isLaneDepartLeft = group['lanepos'] < -(LANEWIDTH/2.0-CARWIDTH/2.0)
    isLaneDepartRight = group['lanepos'] > (LANEWIDTH/2.0-CARWIDTH/2.0)
    isLaneDepart = np.logical_or(isLaneDepartLeft, isLaneDepartRight)
    le, te = find_edges(isLaneDepart)
    num_lane_departs = len(le)

    group.loc[isLaneDepart, 'lanepos'] = np.nan
    group = group.assign(laneseg=pd.Series(np.zeros_like(group['lanepos'])).values)
    le, te = find_edges(group['lanepos'].isnull())
    for i in range(len(le)):
        group.loc[le[i]:, 'laneseg'] += 1
        group.loc[te[i]:, 'laneseg'] += 1

    grouped = group.groupby(['laneseg', 'segment'])
    df_bysegment = grouped.apply(measures_by_segment)
    df_bysegment = df_bysegment.assign(numdeparts=pd.Series(
        np.full_like(df_bysegment['eventid'], num_lane_departs)).values)

    return df_bysegment


def measures_by_segment(group):
    group.reset_index(inplace=True, drop=True)
    df = group.describe()
    data = {'eventid': df['eventid']['max'],
            'ord': df['ord']['max'],
            'time_bin': df['time_bin']['max'],
            'speed_std': df['canspeed']['std'],
            'speed_25%': df['canspeed']['25%'],
            'speed_50%': df['canspeed']['50%'],
            'speed_75%': df['canspeed']['75%'],
            'gyro_z_std': df['gyro_z']['std'],
            'gyro_z_25%': df['gyro_z']['25%'],
            'gyro_z_50%': df['gyro_z']['50%'],
            'gyro_z_75%': df['gyro_z']['75%'],
            'lanepos_std': df['lanepos']['std'],
            'lanepos_25%': df['lanepos']['25%'],
            'lanepos_50%': df['lanepos']['50%'],
            'lanepos_75%': df['lanepos']['75%'],
            'headway_std': df['lv_headway']['std'],
            'headway_25%': df['lv_headway']['25%'],
            'headway_50%': df['lv_headway']['50%'],
            'headway_75%': df['lv_headway']['75%']}

    # canspeed and lanepos are our 'canary in a coal mine' variables
    # canspeed was set to NaN for low speeds and turns in processfile.py
    # lanepos was set to NaN for lane changes and lane departures
    isMissingSpeed = group['canspeed'].isnull()
    isMissingLanepos = group['lanepos'].isnull()
    isMissing = np.logical_or(isMissingSpeed, isMissingLanepos)
    num_nans = len(np.where(isMissing)[0])
    num_tot = df['canspeed']['count']
    if num_tot == 0:
        # print 'by segment: no data in this segment, returning prematurely'
        return
    ratio_nans = float(num_nans)/float(num_tot)
    if ratio_nans > 0.5:
        # print('by segment: too many nans in this segment, ' +
        #       'returning prematurely')
        return

    df_measures = pd.DataFrame(data=data, index=[1])
    df_measures['type'] = group['type'][0]
    df_measures['state'] = group['state'][0]
    df_measures['duration'] = group['duration'][0]

    df_lane = group[['dist_cm', 'lanepos']]
    df_lane.dropna(axis='rows', how='any', inplace=True)
    if len(df_lane) < SEGLENGTH / FS_TIMESTAMP * FS / 3.0:
        df_measures['laneslope'] = np.nan
        df_measures['laneintercept'] = np.nan
    else:
        model = linregress(df_lane['dist_cm']/FS_TIMESTAMP, df_lane['lanepos'])
        df_measures['laneslope'] = model.slope
        df_measures['laneintercept'] = model.intercept

    return df_measures


def find_edges(x):
    ''' find the indices of all the leading and trailing edges of a bool
        array '''
    x = np.array(x)
    shift = x
    shift = np.delete(shift, -1)
    shift = np.insert(shift, 1, shift[0])

    le = np.logical_and(x == 1, shift == 0)
    te = np.logical_and(x == 0, shift == 1)
    idx_le = np.where(le)[0]
    idx_te = np.where(te)[0]

    if np.logical_and(idx_le.size == 0, idx_te.size == 0):
        if any(x):
            idx_le = np.array([1])
            idx_te = np.array([len(x)-1])
        else:
            idx_le = np.array([])
            idx_te = np.array([])
    elif np.logical_and(idx_le.size == 0, idx_te.size > 0):
        idx_le = np.array([1])
    elif np.logical_and(idx_le.size > 0, idx_te.size == 0):
        idx_te = np.array([len(x)-1])
    else:
        if idx_le[0] > idx_te[0]:
            idx_le = np.insert(idx_le, 0, 0)
        if idx_le[-1] > idx_te[-1]:
            idx_te = np.insert(idx_te, len(idx_te), len(x)-1)
    return idx_le, idx_te


if __name__ == '__main__':
    # df2 = pd.DataFrame(np.random.randn(10, 5))
    # outfile = os.path.join('C:\\NADS\\github\\DROW','test.csv')
    # df2.to_csv(outfile)
    df = combinefiles()
