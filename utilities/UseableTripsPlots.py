import numpy as np
import matplotlib.pyplot as plt

y = [511, 511, 532, 511, 533, 1513]

objects = ['lane_width', 'left_line_right_distance', 'left_marker_probability',
           'right_line_left_distance', 'right_marker_probability', 'Non-Zero']

y_pos = np.arange(len(objects))

plt.bar(y_pos, y, align='center')
plt.xticks(y_pos, objects, rotation=45)
plt.ylabel('Trip Count')

plt.title('Useable Files/Trips for Machine Vision Variables')
plt.tight_layout()

plt.show()
