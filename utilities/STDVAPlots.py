import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('H:\VarOneSec.csv', error_bad_lines=False)

x = df['larger_probability']

y = df['vtti.lane_distance_off_center']

plt.scatter(x=x, y=y)
plt.xlabel('Lane Probability')
plt.ylabel('Standard Deviation of Lane Off Center')
plt.title('Lane Marker Probability and Standard Deviation of Lane Off Center')
plt.show()
