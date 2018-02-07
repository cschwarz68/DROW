import pandas as pd
import glob

filenames = glob.glob('H:\Test1LeftRange(1000+)\*.csv')
column = []

for filename in filenames:
    df = pd.read_csv(filename, usecols=['vtti.lane_distance_off_center'],
                     error_bad_lines=False)
    a = df['vtti.lane_distance_off_center']
    b = df['vtti.right_marker_probability']
    column.append(a)
    column.append(b)

new_df = pd.DataFrame([i for i in column])
b = new_df.T
b.to_csv('H:\LeftTimeHist3.csv')
