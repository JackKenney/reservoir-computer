# -*- coding: utf-8 -*-
"""
A minimalistic Echo State Networks demo with Mackey-Glass (delay 17) data 
in "plain" scientific Python.
by Mantas Lukoševičius 2012
http://minds.jacobs-university.de/mantas

Modified and adapted for Sine Wave Synthesis using a Small-World Architecture by: 
by Jack Kenney 2018.
BINDS Lab, Computer Science Department, UMass Amherst
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import networkx as nx

# hyper parameters
in_size = 2  # input vector size
res_size = 350  # size of reservoir
out_size = 1  # output vector size
leak = 0.15  # leaking rate for integrator neurons
noise_mag = 1e-9  # magintude of noise term
aux = 1  # a_0 for x^0  # doing a linear fit
feedback_scaling = 1.5  # output feedback scaling

# load the data
data = np.load('data/waves_500_1000.npy')
in_data = np.asarray(data[0])
# normalize input data
in_min = np.min(in_data)
in_range = np.max(in_data) - in_min
in_data = (in_data - in_min) / in_range
out_data = np.asarray(data[1])
init_len = data[2]
train_len = data[3]
test_len = data[4]

# plot some of the data
plt.figure(1).clear()
plt.plot(in_data[0:1000, 0], 'r')
plt.plot(in_data[0:1000, 1], 'g')
plt.plot(out_data[0:1000], 'b')
plt.title('A sample of data')
plt.legend(['Input freq', 'Input duration', 'Expected output signal'])
# plt.show()

np.random.seed(42)
# initialize random weight matrices
Win = np.random.uniform(-1, 1, (res_size, 1 + in_size))
Wfb = np.random.uniform(-1, 1, (res_size, out_size))

print('Building small world...', end='')
G = nx.connected_watts_strogatz_graph(res_size, 4, 0.7, seed=13)
W = nx.to_numpy_matrix(G)
for i in W:
    for j in i:
        # assign rand to edge (each edge is init_len to 1)
        j *= np.random.uniform(-1, 1)
print('done.')

# save/plot reservoir graph picture
print('Printing graph...', end='')
plt.figure(0).clear()
nx.draw(G)
plt.savefig("last_internal_weight_graph.png")
print('done.')


def prop(u, x, y): 
    """
    This is a function used to propagate 
    input data through the reservoir.
    """
    kept = (1 - leak) * x
    vector = np.dot(
        Win, np.vstack((aux, u))) + \
        np.dot(W, x) + \
        feedback_scaling * np.dot(Wfb, y)
    # TANH activated
    injected = leak * (np.tanh(vector) + noise)
    return kept + injected


# TRAIN
print('Beginning training...', end='')
# allocated memory for the design (collected states) matrix
X = np.zeros((1 + in_size + res_size, train_len - init_len))
# set the corresponding target matrix directly
Yt = out_data[None, init_len + 1:train_len + 1]
# run the reservoir with the data and collect X
x = np.zeros((res_size, 1))

for t in range(train_len):
    noise = noise_mag * np.random.uniform(
        -1, 1)  # small noise term referenced on p.26 of ESN Tutorial
    u = in_data[t]  # Regularize inputs
    y = out_data[t]
    kept = (1 - leak) * x
    x = prop(u, x, y)
    if t >= init_len:
        X[:, t - init_len] = np.vstack((aux, u, x)
                                       )[:, 0].reshape(1 + in_size + res_size)

# batch train the output weights
reg = 1e-8  # regularization coefficient
X_T = X.T
Wout = np.dot(np.dot(Yt, X_T),
              linalg.inv(np.dot(X, X_T) + \
              reg * np.eye(1 + in_size + res_size)))
# optionally use the pseudoinverse - slower
# Wout = np.dot( Yt, linalg.pinv(X) )
print('done.')

# TEST
print('Beginning testing...', end='')
Y = np.zeros((out_size, test_len - 1))
u = in_data[train_len]  # Regularize inputs
y = [[0] * out_size]
# noise only required in training
noise = 0
# run test
for t in range(test_len - 1):
    x = prop(u, x, y)
    y = np.dot(Wout, np.vstack((aux, u, x)))
    Y[:, t] = y
    u = in_data[train_len + t + 1]  # Regularize inputs
print('done.')

# Compute MSE for the first errorLen time steps
errorLen = 500
end = train_len + errorLen + 1
mse = sum(np.square(out_data[train_len + 1:end] - Y[0, 0:errorLen])) / errorLen
print('MSE = ' + str(mse))
rmse = np.sqrt(mse)
nrmse = rmse / (max(out_data) - min(out_data))  # rmse / (ymax-ymin)
print('NRMSE = ' + str(nrmse))

# plt.plot some signals
plt.figure(11).clear()
plt.plot(out_data[train_len + 1:train_len + test_len + 1], 'g')
plt.plot(Y.T, 'b')
plt.title('Target and generated signals')
plt.legend(['Target signal', 'Free-running predicted signal'])

plt.figure(12).clear()
plt.plot(X[0:res_size, 0:200].T)
plt.title('Some reservoir activations')

plt.figure(13).clear()
plt.bar(range(1 + in_size + res_size), Wout.T.reshape(1 + in_size + res_size,))
plt.title('Output weights $\mathbf{W}^{out}$')

plt.show()
