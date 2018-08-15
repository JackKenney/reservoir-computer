"""
Data Generation for Sine Wave Synthesis.
Jack Kenney 2018.
BINDS Lab, Computer Science Department, UMass Amherst.
"""

import numpy as np
import matplotlib.pyplot as plt
import os.path

training_sets = 499
test_sets = 1
freq_order = 10
duration_order = 10
quality = 1000
sets = int(training_sets + test_sets)
file_name = 'data/waves_' + str(sets) + '_' + str(quality)
graphs = True

# Randomization Housekeeping
np.random.seed(7)
# Loop Variables
input_data = []
output_data = []
init_len = 0
train_len = 0
test_len = 0
timekeeper = 0
for count in range(sets):
    # choose random duration & frequency
    freq = np.random.rand() * freq_order  # hertz
    duration = np.random.rand() * duration_order  # seconds
    points = int(quality * duration)
    input_values = [[[freq], [duration]]] * points # (points, 2)
    # generate x values such that there are quality points per second
    x_values = np.array(range(0, points)) / quality
    # calculate corr. y values based on freq.
    y_values = np.sin(x_values * freq * 2 * np.pi).tolist()
    input_data += input_values
    output_data += y_values
    # Determine data lengths
    timekeeper += points #+ 1
    if count == 0:
        init_len = timekeeper
    elif count == training_sets-1:  # done with training data
        train_len = timekeeper
    elif count == sets-test_sets:
        test_len = timekeeper - train_len

if graphs:
    graphdata = np.reshape(input_data[-test_len:], (test_len, 2,))
    plt.figure(1).clear()
    plt.plot(range(test_len), graphdata)
    plt.title('testing input data')

    plt.figure(2).clear()
    plt.plot(range(test_len), output_data[-test_len:])
    plt.title('testing output data')

    plt.show()

output_data = np.array([
    input_data, output_data, init_len, train_len, test_len
])

np.save(file_name + '.npy', output_data)
