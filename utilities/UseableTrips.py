import pandas as pd
import glob

filenames = glob.glob('Y:\TimeseriesExport\*.csv')

left_line_right_distance = 0
right_line_left_distance = 0
left_marker_probability = 0
right_marker_probability = 0
distance_off_center = 0
lane_width = 0

for f in filenames:
    print(f)
    df = pd.read_csv(f, usecols=['vtti.left_line_right_distance',
                                 'vtti.right_line_left_distance',
                                 'vtti.left_marker_probability',
                                 'vtti.right_marker_probability',
                                 'vtti.lane_distance_off_center',
                                 'vtti.lane_width'],
                     error_bad_lines=False).dropna()

    zeros = (df == 0).all()

    if zeros[0]:
        left_line_right_distance += 1
    if zeros[1]:
        right_line_left_distance += 1
    if zeros[2]:
        left_marker_probability += 1
    if zeros[3]:
        right_marker_probability += 1
    if zeros[4]:
        distance_off_center += 1
    if zeros[5]:
        lane_width += 1
