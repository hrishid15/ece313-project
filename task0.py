import scipy.io as sio
import matplotlib.pyplot as plt

# load MAT file
matFile = sio.loadmat('patient_data.mat')

# create subplots and format
figure, axis = plt.subplots(3, figsize=(16, 8))
figure.tight_layout(pad=3.0)

# plot data on each subplot
axis[0].plot(matFile['data'][0])
axis[0].set_title('Heart Rate')
axis[0].set(ylabel='BPM')
axis[1].plot(matFile['data'][1])
axis[1].set_title('Pulse Rate')
axis[1].set(ylabel='BPM')
axis[2].plot(matFile['data'][2])
axis[2].set_title('Respiration Rate')
axis[2].set(ylabel='Breaths/Min')

plt.show()