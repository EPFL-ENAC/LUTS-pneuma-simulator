import numpy as np

# Centimeters in inches
cm = 1 / 2.54
# Speed conversion factor
factor = 3.6
# Vehicle lengths
c_l = 4.4  # car
m_l = 1.6  # moto
# Braking constants
A = 9
B = 1 / 3
# Manoeuvring constants
CM = 4.026397
XM = -0.062306
adaptation_time = 0.6
# Simulation parameters
T = 180  # duration in seconds
L = 90  # road length in meters
d_max = 40.0  # horizon
scaling = 1.5  # lateral scaling
clearance = 0.3  # dart spacing
k = 300  # number of darts thrown
dt = 0.12  # time step in seconds
sqrtdt = np.sqrt(dt)
COUNT = int(T / dt) + 1
# Background grid settings
grid = 0.45
cell = 5 * grid
lane = 8 * grid
lanes = 2

x = np.arange(-L / 2 + grid / 2, L / 2, grid)
y = np.arange(-lane - grid / 2, lane + grid, grid)
xv, yv = np.meshgrid(x, y)
yv = np.flip(yv)
shape = yv.shape

# Results parameters
keep = 1 / 3
