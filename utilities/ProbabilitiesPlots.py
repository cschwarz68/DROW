import numpy as np
import matplotlib.pyplot as plt


y = [26.38936452, 6.27272802, 1.455914131, 1.431837512, 1.069555255,
     0.948294307, 0.92681878, 0.901977953, 0.943526993, 1.052327519,
     58.60765501]

objects = ['0', '1', '100', '200', '300', '400', '500', '600', '700', '800',
           '900+']
y_pos = np.arange(len(objects))

plt.bar(y_pos, y, align='center')
plt.xticks(y_pos, objects)
plt.ylabel('Relative Frequency %')
plt.xlabel('Right Marker Lane Probability Value')
plt.title('Relative Frequencies of Right Marker Lane Probabilies')
plt.show()
