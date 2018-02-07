import pandas as pd
import numpy as np
import glob

filenames = glob.glob('H:\Practice\*.csv')


# replace areas with probabilities less than 800 with nans
for f in filenames:
    print(f)
    df = pd.read_csv(f, error_bad_lines=False)

    df.loc[df['vtti.right_marker_probability'] < 800,
           ['vtti.lane_distance_off_center',
            'vtti.lane_width', 'vtti.left_line_right_distance',
            'vtti.right_line_left_distance']] = np.nan

    df.loc[df['vtti.left_marker_probability'] < 800,
           ['vtti.lane_distance_off_center', 'vtti.lane_width',
            'vtti.left_line_right_distance',
            'vtti.right_line_left_distance']] = np.nan

    # fill segments with less than 1 second of consecutive nan's with
    # linear interpolates
    df['vtti.lane_distance_off_center'].interpolate(method='linear', limit=10)
    df['vtti.left_line_right_distance'].interpolate(method='linear', limit=10)
    df['vtti.right_line_left_distance'].interpolate(method='linear', limit=10)
    df['vtti.lane_width'].interpolate(method='linear', limit=10)
