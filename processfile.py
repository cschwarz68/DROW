import os
import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import mode
import matplotlib.pyplot as plt

KPH2MPS = 1/3.6
G2MPSS = 9.8
FS = 10
FS_TIMESTAMP = 1000
LATOFFSET = 2
INTERPLIM = 10
PREVIEW_SECONDS = 240
LANEPROBLIM = 800
MAKEPLOTS = True
WRITEFILE = True


def process(fullfile, df_ord):
    '''
    Read a 10 hz file

    Returns:
    A processed data frame from the trip
    '''

    drive, path = os.path.splitdrive(fullfile)
    path, filename = os.path.split(path)
    filename, file_extension = os.path.splitext(filename)
    print('processing ' + filename)

    try:
        df = pd.read_csv(fullfile,
                         usecols=['vtti.timestamp', 'computed.time_bin',
                                  'vtti.speed_network', 'vtti.accel_x',
                                  'vtti.accel_y', 'vtti.gyro_z',
                                  'vtti.left_line_right_distance',
                                  'vtti.right_line_left_distance',
                                  'vtti.left_marker_probability',
                                  'vtti.right_marker_probability',
                                  'vtti.lane_distance_off_center',
                                  'vtti.lane_width', 'TRACK1_TARGET_ID',
                                  'TRACK2_TARGET_ID', 'TRACK3_TARGET_ID',
                                  'TRACK4_TARGET_ID', 'TRACK5_TARGET_ID',
                                  'TRACK6_TARGET_ID', 'TRACK7_TARGET_ID',
                                  'TRACK8_TARGET_ID', 'TRACK1_X_POS_PROCESSED',
                                  'TRACK2_X_POS_PROCESSED',
                                  'TRACK3_X_POS_PROCESSED',
                                  'TRACK4_X_POS_PROCESSED',
                                  'TRACK5_X_POS_PROCESSED',
                                  'TRACK6_X_POS_PROCESSED',
                                  'TRACK7_X_POS_PROCESSED',
                                  'TRACK8_X_POS_PROCESSED',
                                  'TRACK1_Y_POS_PROCESSED',
                                  'TRACK2_Y_POS_PROCESSED',
                                  'TRACK3_Y_POS_PROCESSED',
                                  'TRACK4_Y_POS_PROCESSED',
                                  'TRACK5_Y_POS_PROCESSED',
                                  'TRACK6_Y_POS_PROCESSED',
                                  'TRACK7_Y_POS_PROCESSED',
                                  'TRACK8_Y_POS_PROCESSED',
                                  'TRACK1_X_VEL_PROCESSED',
                                  'TRACK2_X_VEL_PROCESSED',
                                  'TRACK3_X_VEL_PROCESSED',
                                  'TRACK4_X_VEL_PROCESSED',
                                  'TRACK5_X_VEL_PROCESSED',
                                  'TRACK6_X_VEL_PROCESSED',
                                  'TRACK7_X_VEL_PROCESSED',
                                  'TRACK8_X_VEL_PROCESSED'],
                         error_bad_lines=False)
        df.columns = ['timestamp', 'time_bin', 'canspeed', 'acc_x', 'acc_y',
                      'gyro_z', 'dist_to_left', 'dist_to_right', 'leftprob',
                      'rightprob', 'lanepos', 'lanewidth', 'track1_id',
                      'track2_id', 'track3_id', 'track4_id', 'track5_id',
                      'track6_id', 'track7_id', 'track8_id', 'track1_xpos',
                      'track2_xpos', 'track3_xpos', 'track4_xpos',
                      'track5_xpos', 'track6_xpos', 'track7_xpos',
                      'track8_xpos', 'track1_ypos', 'track2_ypos',
                      'track3_ypos', 'track4_ypos', 'track5_ypos',
                      'track6_ypos', 'track7_ypos', 'track8_ypos',
                      'track1_xvel', 'track2_xvel', 'track3_xvel',
                      'track4_xvel', 'track5_xvel', 'track6_xvel',
                      'track7_xvel', 'track8_xvel']
    except Exception:
        print('read_csv failed on: ' + fullfile)
        return

    # drop lane data when probabilities are low
    isLowProb = df.loc[:, 'leftprob'] + df.loc[:, 'rightprob'] < LANEPROBLIM
    df.loc[isLowProb, 'lanepos'] = np.nan
    df.loc[isLowProb, 'lanewidth'] = np.nan

    # interpolate through dropped frames
    df = interp_data(df)

    # filter acceleration and estimate where turns occur.
    # mark those speeds with NaNs
    isturn = remove_turns(df)
    df.loc[isturn, 'canspeed'] = np.nan

    # set slow speeds to NaNs
    isslow = df.canspeed < 30
    df.loc[isslow, 'canspeed'] = np.nan

    # interpolate through dropped frames
    # second time to fill in short gaps left by last two steps
    df = interp_data(df)

    # trim size of file by getting rid of empty rows, and reset the index
    # df = df[df.canspeed.notnull()]
    # df = df.reset_index(drop=True)

    # test for a short file
    if len(df) < 100:
        return

    if MAKEPLOTS:
        F = plt.figure()
        plt.subplot(321)
        plt.plot(df.timestamp, df.lanepos)
        plt.title('lanepos')
        plt.subplot(322)
        plt.plot(df.timestamp, df.lanewidth)
        plt.title('lane width')
        plt.subplot(323)
        plt.plot(df.timestamp, df.dist_to_left)
        plt.title('left line')
        plt.subplot(324)
        plt.plot(df.timestamp, df.dist_to_right)
        plt.title('right line')
        plt.subplot(325)
        plt.plot(df.timestamp, df.leftprob)
        plt.title('left prob')
        plt.subplot(326)
        plt.plot(df.timestamp, df.rightprob)
        plt.title('right prob')
        F.set_size_inches(20, 20)
        plt.pause(1)
        plt.savefig(os.path.join('laneplots', filename + '.png'))

    if MAKEPLOTS:
        plt.close('all')

    # find headway and range rate for lead vehicle
    # remove the processed track columns
    df = process_radar(df, filename)

    # integrate ORD ratings from df_ord
    dflist = ord_list(df, df_ord, filename)

    if not dflist:
        print('no valid ORD segments have been found')
        return
    else:
        df_trip = pd.concat(dflist, axis=0)

    # save ORD segments to a file
    if WRITEFILE:
        # export dataframe to csv file
        outfile = os.path.join(os.getenv('SHRP2ProcessedII'),
                               filename + '.csv')
        print("saving ORD segments to file " + outfile)
        df_trip.to_csv(outfile, index=None)

    return


def interp_data(df):
    '''
    interpolate through dropped frames up to periods of INTERPLIM frames
    in length.
    '''

    df['canspeed'].interpolate(method='linear', limit=2*INTERPLIM,
                               inplace=True)
    df['gyro_z'].interpolate(method='linear', limit=INTERPLIM, inplace=True)
    df['acc_x'].interpolate(method='linear', limit=INTERPLIM, inplace=True)
    df['acc_y'].interpolate(method='linear', limit=INTERPLIM, inplace=True)
    df['lanepos'].interpolate(method='linear', limit=INTERPLIM, inplace=True)
    df['dist_to_left'].interpolate(method='linear', limit=INTERPLIM,
                                   inplace=True)
    df['dist_to_right'].interpolate(method='linear', limit=INTERPLIM,
                                    inplace=True)
    df['leftprob'].interpolate(method='linear', limit=INTERPLIM, inplace=True)
    df['rightprob'].interpolate(method='linear', limit=INTERPLIM, inplace=True)
    df['lanewidth'].interpolate(method='linear', limit=INTERPLIM, inplace=True)

    return df


def remove_turns(df):
    b1, a1 = signal.butter(4, 0.09/(FS/2.0), 'low', analog=False)
    b2, a2 = signal.butter(4, 1.3/(FS/2.0), 'low', analog=False)

    Ay_low1 = filter_segments(b1, a1, df.acc_y)
    Ay_high1 = df['acc_y'] - Ay_low1
    Ay_low2 = filter_segments(b2, a2, Ay_high1)
    isturn = abs(Ay_low2) > 0.03

    if MAKEPLOTS:
        plt.figure(1)
        plt.plot(df.timestamp, df.acc_y)
        plt.hold(True)
        plt.plot(df.timestamp, Ay_high1, 'g')
        plt.plot(df.timestamp, Ay_low1, 'r')
        plt.plot(df.timestamp, Ay_low2, 'c')
        plt.plot(df.timestamp, isturn*0.2, 'mo')
        plt.pause(1)

    return isturn


def filter_segments(b, a, x):
    x = np.array(x)
    idx_valid_le, idx_valid_te = find_edges(~np.isnan(x))
    for i in range(len(idx_valid_le)):
        x[idx_valid_le[i]:idx_valid_te[i]] = filter(b, a, x[idx_valid_le[i]:
                                                    idx_valid_te[i]])
    return x


def filter(b, a, x):
    if len(x) <= 3*max(len(a), len(b)):
        y = x
        return y

    y = signal.filtfilt(b, a, x)
    return y


def find_edges(x):
    ''' find the indices of all the leading and trailing edges of a bool
        array '''
    shift = x
    mask = np.ones(len(shift), dtype=bool)
    mask[-1] = 0
    shift = shift[mask]
    shift = np.insert(shift, 0, shift[0])

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


def process_radar(df, filename):
    '''
    use the 8 processed radar tracks to estimate which is the lead vehicle,
    and what it's headway and range rate are. It seems like many of the ypos
    values are nans, even if the xpos or id values are not...mystery
    '''

    isOffsetLane1 = abs(df.loc[:, 'track1_ypos']) > LATOFFSET
    isOffsetLane2 = abs(df.loc[:, 'track2_ypos']) > LATOFFSET
    isOffsetLane3 = abs(df.loc[:, 'track3_ypos']) > LATOFFSET
    isOffsetLane4 = abs(df.loc[:, 'track4_ypos']) > LATOFFSET
    isOffsetLane5 = abs(df.loc[:, 'track5_ypos']) > LATOFFSET
    isOffsetLane6 = abs(df.loc[:, 'track6_ypos']) > LATOFFSET
    isOffsetLane7 = abs(df.loc[:, 'track7_ypos']) > LATOFFSET
    isOffsetLane8 = abs(df.loc[:, 'track8_ypos']) > LATOFFSET

    df.loc[isOffsetLane1, 'track1_id'] = np.nan
    df.loc[isOffsetLane2, 'track2_id'] = np.nan
    df.loc[isOffsetLane3, 'track3_id'] = np.nan
    df.loc[isOffsetLane4, 'track4_id'] = np.nan
    df.loc[isOffsetLane5, 'track5_id'] = np.nan
    df.loc[isOffsetLane6, 'track6_id'] = np.nan
    df.loc[isOffsetLane7, 'track7_id'] = np.nan
    df.loc[isOffsetLane8, 'track8_id'] = np.nan

    df.loc[isOffsetLane1, 'track1_ypos'] = np.nan
    df.loc[isOffsetLane2, 'track2_ypos'] = np.nan
    df.loc[isOffsetLane3, 'track3_ypos'] = np.nan
    df.loc[isOffsetLane4, 'track4_ypos'] = np.nan
    df.loc[isOffsetLane5, 'track5_ypos'] = np.nan
    df.loc[isOffsetLane6, 'track6_ypos'] = np.nan
    df.loc[isOffsetLane7, 'track7_ypos'] = np.nan
    df.loc[isOffsetLane8, 'track8_ypos'] = np.nan

    df.loc[isOffsetLane1, 'track1_xpos'] = np.nan
    df.loc[isOffsetLane2, 'track2_xpos'] = np.nan
    df.loc[isOffsetLane3, 'track3_xpos'] = np.nan
    df.loc[isOffsetLane4, 'track4_xpos'] = np.nan
    df.loc[isOffsetLane5, 'track5_xpos'] = np.nan
    df.loc[isOffsetLane6, 'track6_xpos'] = np.nan
    df.loc[isOffsetLane7, 'track7_xpos'] = np.nan
    df.loc[isOffsetLane8, 'track8_xpos'] = np.nan

    df.loc[isOffsetLane1, 'track1_xvel'] = np.nan
    df.loc[isOffsetLane2, 'track2_xvel'] = np.nan
    df.loc[isOffsetLane3, 'track3_xvel'] = np.nan
    df.loc[isOffsetLane4, 'track4_xvel'] = np.nan
    df.loc[isOffsetLane5, 'track5_xvel'] = np.nan
    df.loc[isOffsetLane6, 'track6_xvel'] = np.nan
    df.loc[isOffsetLane7, 'track7_xvel'] = np.nan
    df.loc[isOffsetLane8, 'track8_xvel'] = np.nan

    vhid = df.loc[:, 'track1_id':'track8_id']
    xpos = df.loc[:, 'track1_xpos':'track8_xpos']
    ypos = df.loc[:, 'track1_ypos':'track8_ypos']
    xvel = df.loc[:, 'track1_xvel':'track8_xvel']

    if MAKEPLOTS:
        F = plt.figure()
        plt.subplot(221)
        plt.plot(df.timestamp, vhid)
        plt.title('vhid')
        plt.subplot(222)
        plt.plot(df.timestamp, xpos)
        plt.title('xpos')
        plt.subplot(223)
        plt.plot(df.timestamp, ypos)
        plt.title('ypos')
        plt.subplot(224)
        plt.plot(df.timestamp, xvel)
        plt.title('xvel')
        F.set_size_inches(16, 8)
        plt.pause(1)
        plt.savefig(os.path.join('radarplots', filename + '.png'))

    imin_xpos_label = abs(xpos).idxmin(axis='columns')
    imin_xpos = [int(i[5])-1 if isinstance(i, str) else None for i in
                 imin_xpos_label]
    lv_id = [row[imin_xpos[i]] if imin_xpos[i] else None for (i, row) in
             enumerate(vhid.itertuples())]
    lv_headway = [row[imin_xpos[i]] if imin_xpos[i] else None for (i, row) in
                  enumerate(xpos.itertuples())]
    lv_range_rate = [row[imin_xpos[i]] if imin_xpos[i] else None for (i, row)
                     in enumerate(xvel.itertuples())]

    df['lv_id'] = lv_id
    df['lv_headway'] = lv_headway
    df['lv_range_rate'] = lv_range_rate

    # drop original radar columns, leaving only the new lv variables
    cols = [c for c in df.columns if c[:5] != 'track']
    df = df[cols]

    return df


def ord_list(df, df_ord, filename):
    # file integer ID
    fileid = int(filename[8:])

    # get computed time bin and remove that column
    timebin = mode(df.time_bin).mode[0]

    # look for ORD event(s) in this trip
    dflist = []
    isfile = df_ord.loc[:, 'fileid'] == fileid
    df_ord_trip = df_ord.loc[isfile, :]
    for row in df_ord_trip.iterrows():
        start_time = int(row[1].start_time)
        start_time = start_time - (PREVIEW_SECONDS * FS_TIMESTAMP)
        end_time = int(row[1].end_time)
        eventid = int(row[1].eventid)
        if row[1].ord == '.':
            continue
        ordval = int(row[1].ord)
        df_segment = df.loc[np.logical_and(df.timestamp >= start_time,
                            df.timestamp <= end_time), :]
        if len(df_segment) == 0:
            continue
        df_segment.loc[:, 'timebin'] = timebin
        df_segment['eventid'] = eventid
        df_segment['ord'] = ordval
        df_segment['type'] = row[1].type
        df_segment['state'] = row[1].coded_state
        dflist.append(df_segment)

    return dflist


if __name__ == '__main__':
    # read in the ORD data
    ordfile = 'Final_ORDratings_2-14-17_wTrip+Timestamp.csv'
    df_ord = pd.read_csv(ordfile,
                         usecols=['File_ID-Trip_ID', 'ORD_EVENT_ID',
                                  'ORD_Rating_Starttime',
                                  'ORD_Rating_Endtime', 'AVEORDRATING',
                                  'Event_Type', 'SampleSource'],
                         error_bad_lines=False)
    df_ord.columns = ['fileid', 'eventid', 'start_time', 'end_time', 'ord',
                      'type', 'coded_state']

    # process a file
    filename = os.path.join(os.getenv('SHRP2DataII'), 'TimeSeries_Files',
                            'File_ID_25087961.csv')
    process(filename, df_ord)

    # process a file
    filename = os.path.join(os.getenv('SHRP2DataII'), 'TimeSeries_Files',
                            'File_ID_1093845.csv')
    process(filename, df_ord)

    # process a file
    filename = os.path.join(os.getenv('SHRP2DataII'), 'TimeSeries_Files',
                            'File_ID_727784.csv')
    process(filename, df_ord)
