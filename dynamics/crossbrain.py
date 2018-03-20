import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

iterations = 10000
step_size = 0.01
number_of_modes = 3
epsilon = 10e-18
environment_fluctuations = np.random.rand(number_of_modes)

x_temporal_patterns = np.random.rand(number_of_modes) # x
sigma = [1, 1.1, 0.9] # excitation rate per mode
q = 0.10 # coupling to y
roe = np.random.rand(number_of_modes, number_of_modes) # within brain cognitive inhibition matrix
roe[0,1] = 1.55*sigma[0]/sigma[1]
roe[0,2] = 0.62*sigma[0]/sigma[2]
roe[1,2] = 1.45*sigma[1]/sigma[2]
roe[1,0] = 0.60*sigma[1]/sigma[0]
roe[2,0] = 1.65*sigma[2]/sigma[0]
roe[2,1] = 0.70*sigma[2]/sigma[1]
theta = np.random.rand(number_of_modes, number_of_modes) # between brain cognitive inhibition matrix
def x_environment_fluctuations(i, x):
    return 0

y_temporal_patterns = np.random.rand(number_of_modes) # y
delta = [2.2, 2.1, 1.9] # excitation rate per mode
p = 0.22 # coupling to x
xi = np.random.rand(number_of_modes, number_of_modes) # within brain cognitive inhibition matrix
xi[0,1] = 1.55*delta[0]/delta[1]
xi[0,2] = 0.62*delta[0]/delta[2]
xi[1,2] = 1.45*delta[1]/delta[2]
xi[1,0] = 0.60*delta[1]/delta[0]
xi[2,0] = 1.65*delta[2]/delta[0]
xi[2,1] = 0.70*delta[2]/delta[1]
def nu(i, j): # between brain cognitive inhibition matrix
    return i+1 + 0.2*(j+1)*(j+1)
def y_environment_fluctuations(i, x):
    return 0

def dx_i(i, x):
    within_brain = [roe[i,j]*x[j] for j in range(number_of_modes) if j != i]
    other_brain = [theta[i,s]*y_temporal_patterns[s] for s in range(number_of_modes)]
    return x[i]*(sigma[i]
                 -x[i]
                 -sum(within_brain)
                 -q * sum(other_brain)) + epsilon * x_environment_fluctuations(i, x[i])

def dx(x):
    return np.array([dx_i(i, x) for i in range(number_of_modes)])

def dy_i(i, y):
    within_brain = [xi[i,j]*y[j] for j in range(number_of_modes) if j != i]
    other_brain = [nu(i,s)*x_temporal_patterns[s] for s in range(number_of_modes)]
    return y[i]*(delta[i]
                 -y[i]
                 -sum(within_brain)
                 -p * sum(other_brain)) + epsilon * y_environment_fluctuations(i, y[i])

def dy(y):
    return np.array([dy_i(i, y) for i in range(number_of_modes)])

# Returns approximate change in x
def runge_kutta(x, derivative, step_size):
    k1 = derivative(x)*step_size
    k2 = derivative(x + k1/2)*step_size
    k3 = derivative(x + k2/2)*step_size
    k4 = derivative(x + k3)*step_size
    return (k1+2*k2+2*k3+k4)/6

def update(x, y, step_size):
    change_x = runge_kutta(x_temporal_patterns, dx, step_size)
    change_y = runge_kutta(y_temporal_patterns, dy, step_size)
    return change_x, change_y, x+change_x, y+change_y

x1, x2, x3 = np.zeros((iterations)), np.zeros((iterations)), np.zeros((iterations))
y1, y2, y3 = np.zeros((iterations)), np.zeros((iterations)), np.zeros((iterations))
t = np.zeros((iterations))
for i in range(iterations):
    c_x, c_y, x_temporal_patterns, y_temporal_patterns = update(x_temporal_patterns, y_temporal_patterns, step_size)
    x1[i] = x_temporal_patterns[0]
    x2[i] = x_temporal_patterns[1]
    x3[i] = x_temporal_patterns[2]

    y1[i] = y_temporal_patterns[0]
    y2[i] = y_temporal_patterns[1]
    y3[i] = y_temporal_patterns[2]
    t[i] = i*step_size

plt.subplot(211)
plt.plot(t, x1, color='red')
plt.plot(t, x2, color='green')
plt.plot(t, x3, color='blue')

plt.subplot(212)
plt.plot(t, x1, color='red')
plt.plot(t, x2, color='green')
plt.plot(t, x3, color='blue')
plt.show()
