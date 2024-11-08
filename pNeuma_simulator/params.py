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

car_l = 2.20
car_w = 0.90
car_a = 0.17
car_b = 3.50

moto_l = 0.80
moto_w = 0.30
moto_a = 0.25
moto_b = 1.30

car_args = [
    (4.60, 0, 1.30),
    (0.20, 0, 29.0),
    (1.30, 5.80, 0, 2.10),
]
car_bounds = [
    (0.80, 3.50),
    (22.1, 47.6),
    (0.40, 3.60),
]
moto_args = [
    (2.40, 0, 2.80),
    (0.30, 0, 33.1),
    (0.90, 2.50, 0, 2.20),
]
moto_bounds = [
    (1.10, 7.00),
    (23.9, 46.0),
    (0.20, 3.40),
]
