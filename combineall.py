import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import linregress

FS = 10
FS_TIMESTAMP = 1000
SEGLENGTH = 6 * FS_TIMESTAMP

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
                         usecols=['timestamp', 'time_bin', 'canspeed', 'acc_x',
                                  'acc_y', 'gyro_z', 'lanepos', 'lanewidth',
                                  'lv_id', 'lv_headway', 'lv_range_rate',
                                  'eventid', 'ord', 'type', 'state'],
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
        outfile = os.path.join('C:\\NADS\\github\\DROW', 'combined.csv')
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
    if float(num_nans)/float(num_tot) > 0.9:
        print('by event: too many nans, returning prematurely')
        return
    group['dist_cm'] = group['canspeed_cmps'].cumsum() / FS

    grouped = group.groupby('segment')
    df_bysegment = grouped.apply(measures_by_segment)

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

    # canspeed is our 'canary in a coal mine'
    # it has been set to NaN for low speeds and turns in processfile.py
    num_nans = len(np.where(np.isnan(group['canspeed']))[0])
    num_tot = df['canspeed']['count']
    if num_tot == 0:
        # print 'by segment: no data in this segment, returning prematurely'
        return
    ratio_nans = float(num_nans)/float(num_tot)
    if ratio_nans > 0.5:
        # print('by segment: too many nans in this segment, ' +
        # 'returning prematurely')
        return

    df_measures = pd.DataFrame(data=data, index=[1])
    df_measures['type'] = group['type'][0]
    df_measures['state'] = group['state'][0]

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


if __name__ == '__main__':
    # df2 = pd.DataFrame(np.random.randn(10, 5))
    # outfile = os.path.join('C:\\NADS\\github\\DROW','test.csv')
    # df2.to_csv(outfile)
    df = combinefiles()
