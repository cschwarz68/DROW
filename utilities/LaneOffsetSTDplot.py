import numpy as np
import matplotlib.pyplot as plt

y = [29.96821172, 9.57406813, 12.48006565, 10.16290276, 7.731386949,
     5.754940334, 4.236148929, 3.117784088, 2.340054346, 1.784953888,
     12.84948321]

objects = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10+']
y_pos = np.arange(len(objects))

plt.bar(y_pos, y, align='center')
plt.xticks(y_pos, objects)
plt.ylabel('Relative Frequency %')
plt.xlabel('One Second Standard Deviation')
plt.title('Relative Frequency of Standard Deviations for Lane Offset')
plt.show()
