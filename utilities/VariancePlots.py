import pandas as pd
import glob

path = r'C:\TimeSeriesExport'
filenames = glob.glob(path + "\*.csv")
file_list = []

# concatenate the entire dataset, dropping nAn values
for f in filenames:
    df = pd.read_csv(f, usecols=['vtti.left_marker_probability',
                                 'vtti.right_marker_probability',
                                 'vtti.lane_distance_off_center'],
                     error_bad_lines=False).dropna()
    file_list.append(df)
    print(f)
frame = pd.concat(file_list)

frame2 = frame.loc['vtti.left_marker_probability',
                   'vtti.right_marker_probability']
frame2.to_csv('H:\Giantframe.csv')
