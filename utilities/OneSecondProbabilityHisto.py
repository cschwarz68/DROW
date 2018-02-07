import pandas as pd
import numpy as np

df = pd.read_csv('H:\GiantFrame.csv', error_bad_lines=False)

x = df['vtti.left_marker_probability']
x = list(x.values)

bins = [0, 1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1024]
hist, bins = np.histogram(x, bins=bins)
