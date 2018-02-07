import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('H:\Test1LeftRange(400-600)\File_ID_59324053.csv',
                 error_bad_lines=False)

a = df['vtti.left_marker_probability']
b = df['vtti.right_marker_probability']
c = df['vtti.lane_distance_off_center']
d = df['vtti.timestamp']
xmin = d.min()
xmax = d.max()
ymin = c.min() * 1.05
ymax = 1100

plt.plot(a, label='Left Marker Lane Probability')
plt.plot(b, label='Right Marker Lane Probability')
plt.plot(c, label='Lane Off Center')
plt.xticks(d)

plt.xlabel('Camera Timestamp')
plt.title('Time History of Trip ID #59324053')
# plt.legend(loc = 4, prop={'size':8})

plt.show()
