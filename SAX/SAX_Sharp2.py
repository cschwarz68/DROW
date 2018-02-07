import numpy as np
import pandas as pd
import glob

filenames = glob.glob("Y:\TimeseriesExport\*.csv")


def main():
    for filename in filenames:

        print(filename)

        df = pd.read_csv(filename, usecols=['vtti.file_id',
                                            'vtti.speed_network',
                                            'vtti.accel_x', 'vtti.accel_y',
                                            'vtti.gyro_z',
                                            'vtti.lane_distance_off_center',
                                            'vtti.speed_gps', 'vtti.prndl',
                                            'TRACK1_TARGET_ID',
                                            'computed.time_bin',
                                            'vtti.alcohol_interior'])

        # check to see if the trip has valid speed
        if no_speed(df):
            continue
        # trim the trip based on speed
        df = trim_file(df)

        # revise the column names
        col_names = [col for col in df]
        for index in range(len(col_names)):
            if 'vtti.' in col_names[index]:
                col_names[index] = col_names[index].replace("vtti.", "")
        df.columns = col_names

        trip_id = filename[28:-4]

        # make 10hz data into 1hz
        df = df.groupby(lambda x: x/10).mean()

        # replace lead vehicle ID with true/false
        lead_ID = df.TRACK1_TARGET_ID
        lead_vehicle = []

        for num in lead_ID:
            if num > 0:
                lead_vehicle.append('T')
            else:
                lead_vehicle.append('F')
        df['lead_vehicle'] = lead_vehicle
        df = df.drop('TRACK1_TARGET_ID', 1)

        # fill nan value in spd_net
        spd_net = np.array(df['speed_network'])
        spd_gps = np.array(df['speed_gps'])
        nan_index = np.isnan(spd_net)
        spd_net[nan_index] = spd_gps[nan_index]
        df['speed_network'] = spd_net

        for index in range(len(spd_net)):
            if spd_net[index] != spd_net[index]:
                spd_net[index] = (spd_net[index-1]+spd_net[index+1])/2

        # reset variable time_bin and prndl
        time_bin = np.array(df['computed.time_bin'])
        for index in range(len(time_bin)):
            if time_bin[index] == time_bin[index]:
                df['computed.time_bin'] = time_bin[index]
                break

        # filt gear position with deleting rows with parking, reversing,
        # natural and invalid data
        prndl = np.array(df['prndl'])
        df['prndl'] = np.around(prndl)
        prndl_noneed_rows = []

        for index in range(len(prndl)):
            if prndl[index] == 0 or prndl[index] == 1 or prndl[index] == 2 or prndl[index] >= 5:
                prndl_noneed_rows.append(index)
        df = df.drop(df.index[[prndl_noneed_rows]])

        # drop variable 'gear position' and 'speed_gps' after finishing filter
        df = df.drop('prndl', 1)
        df = df.drop('speed_gps', 1)

        # convert value of speed and acc into letters
        df = speed_ltr(df)
        df = acc_x_ltr(df)
        df = acc_y_ltr(df)

        df.to_csv("H:\SHRP2\SAX_Sharp2\SAX_Sharp2_"+trip_id+".csv", index=None)


def no_speed(df):
    ''' Reject a file if there is no movement '''
    if not(any(pd.notnull(df['vtti.speed_network']))):
        print("speed no values")
        return True
    if max(df['vtti.speed_network'][pd.notnull(df['vtti.speed_network'])]) == 0:
        print("vehicle not moving")
        return True
    return False


def trim_file(df):
    ''' Trim the beginning and end of a file based on speed '''
    ismoving = df['vtti.speed_network'] > 0
    idx_first = np.where(ismoving)[0][0]
    idx_last = np.where(ismoving)[0][-1]
    try:
        df = df[idx_first:idx_last+1]
    except Exception:
        df = df[idx_first:idx_last]
    return df


def speed_ltr(df):
    '''convert speed value into letters'''
    speed_ltr = []
    for num in np.array(df.speed_network):
        if num == 0:
            speed_ltr.append('a')
        elif num > 0 and num <= 16:
            speed_ltr.append('b')
        elif num > 16 and num <= 32:
            speed_ltr.append('c')
        elif num > 32 and num <= 48:
            speed_ltr.append('d')
        elif num > 48 and num <= 64:
            speed_ltr.append('e')
        elif num > 64 and num <= 80:
            speed_ltr.append('f')
        elif num > 80 and num <= 97:
            speed_ltr.append('g')
        elif num > 97 and num <= 113:
            speed_ltr.append('h')
        elif num > 113:
            speed_ltr.append('i')
        else:
            speed_ltr.append('')
    df['speed_ltr'] = speed_ltr
    return df


def acc_x_ltr(df):
    '''convert acc_x value into letters'''
    acc_x_ltr = []
    for num in np.array(df.accel_x):
        if num <= -0.6:
            acc_x_ltr.append('a')
        elif num > -0.6 and num <= -0.4:
            acc_x_ltr.append('b')
        elif num > -0.4 and num <= -0.3:
            acc_x_ltr.append('c')
        elif num > -0.3 and num <= -0.05:
            acc_x_ltr.append('d')
        elif num > -0.05 and num <= 0:
            acc_x_ltr.append('e')
        elif num > 0 and num <= 0.05:
            acc_x_ltr.append('f')
        elif num > 0.05 and num <= 0.3:
            acc_x_ltr.append('g')
        elif num > 0.3 and num <= 0.4:
            acc_x_ltr.append('h')
        elif num > 0.4 and num <= 0.6:
            acc_x_ltr.append('i')
        elif num > 0.6:
            acc_x_ltr.append('j')
        else:
            acc_x_ltr.append('')
    df['acc_x_ltr'] = acc_x_ltr
    return df


def acc_y_ltr(df):
    '''convert acc_y values into letters'''
    acc_y_ltr = []
    for num in np.array(df.accel_y):
        if num <= -0.6:
            acc_y_ltr.append('a')
        elif num > -0.6 and num <= -0.4:
            acc_y_ltr.append('b')
        elif num > -0.4 and num <= -0.3:
            acc_y_ltr.append('c')
        elif num > -0.3 and num <= -0.05:
            acc_y_ltr.append('d')
        elif num > -0.05 and num <= 0:
            acc_y_ltr.append('e')
        elif num > 0 and num <= 0.05:
            acc_y_ltr.append('f')
        elif num > 0.05 and num <= 0.3:
            acc_y_ltr.append('g')
        elif num > 0.3 and num <= 0.4:
            acc_y_ltr.append('h')
        elif num > 0.4 and num <= 0.6:
            acc_y_ltr.append('i')
        elif num > 0.6:
            acc_y_ltr.append('j')
        else:
            acc_y_ltr.append('')
    df['acc_y_ltr'] = acc_y_ltr
    return df


main()
