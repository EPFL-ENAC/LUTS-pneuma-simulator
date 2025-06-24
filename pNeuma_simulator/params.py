import numpy as np

# Centimeters in inches
cm = 1 / 2.54
# Speed conversion factor
factor = 3.6

# Simulation parameters
T = 180  # duration in seconds
L = 90  # road length in meters
d_max = 40.0  # horizon
clearance = 0.3  # dart spacing
k = 300  # number of darts thrown
da = 0.25  # angular resolution
dt = 0.02  # time step in seconds
sqrtdt = np.sqrt(dt)
COUNT = int(T / dt) + 1

# Background grid settings
lanes = 2
grid = 0.45
cell = 5 * grid
lane = 8 * grid
x = np.arange(-L / 2 + grid / 2, L / 2, grid)
y = np.arange(-lane - grid / 2, lane + grid, grid)
xv, yv = np.meshgrid(x, y)
yv = np.flip(yv)
shape = yv.shape

# Steady state
keep = 1 / 3

# Braking constants
A = 6
B = 1 / 2
# Manoeuvring constants
CM = 4.026397
XM = -0.062306

# Model parameters
tau = 0.6
uptau_min = 0.2
uptau_max = 1.0
scaling = 1.5  # lateral

car_l = 2.20
car_w = 0.90
car_a = 0.17
car_b = 3.80

moto_l = 0.80
moto_w = 0.30
moto_a = 0.24
moto_b = 1.80

car_args = [
    (4.85, 0, 1.30),
    (0.20, 0, 8 * factor),
    (1.50, 5.75, 0, 2.10),
]
car_bounds = [
    (0.80, 3.50),
    (22.5, 50.0),
    (0.30, 3.00),
]
moto_args = [
    (3.00, 0, 2.60),
    (0.25, 0, 9 * factor),
    (1.00, 3.30, 0, 2.00),
]
moto_bounds = [
    (1.00, 7.00),
    (22.5, 50.0),
    (0.10, 3.00),
]
