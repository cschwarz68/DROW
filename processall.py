import glob
import os
import pandas as pd
from processfile import process

PATH = os.getenv('SHRP2DataII')


def processfiles():
    '''
    loop through all the csv files in the SHRP2 path and process each one,
    integrating the ORD ratings during the processing
    '''

    # list of csv files
    files = glob.glob(os.path.join(PATH, 'TimeSeries_Files', 'File_ID_*.csv'))

    # read in the ORD data
    ordfile = 'Final_ORDratings_2-14-17_wTrip+Timestamp.csv'
    df_ord = pd.read_csv(ordfile,
                         usecols=['File_ID-Trip_ID', 'ORD_EVENT_ID',
                                  'ORD_Rating_Starttime', 'ORD_Rating_Endtime',
                                  'AVEORDRATING', 'Event_Type',
                                  'SampleSource'],
                         error_bad_lines=False)
    df_ord.columns = ['fileid', 'eventid', 'start_time', 'end_time', 'ord',
                      'type', 'coded_state']

    # loop through all the csv files and process each one,
    # skipping if the corresponding processed file already exists
    for fullfile in files:
        drive, path = os.path.splitdrive(fullfile)
        path, filename = os.path.split(path)
        filename, file_extension = os.path.splitext(filename)

        ordname = os.path.join(os.getenv('SHRP2ProcessedII'),
                               filename + '.csv')
        if os.path.isfile(ordname):
            print('ORD file exists for file ' + filename)
            continue

        process(fullfile, df_ord)


if __name__ == '__main__':
    processfiles()
