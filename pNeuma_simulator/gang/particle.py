from matplotlib.patches import Ellipse
from numpy import array, atan2, cos, degrees, sin

from pNeuma_simulator import params


class Particle:
    """A class representing a two-dimensional particle."""

    def __init__(self, x, y, vx, vy, mode="Car", ID=None, styles=None):
        """Initialize the particle's position, velocity, mode and ID.

        Any key-value pairs passed in the styles dictionary will be passed
        as arguments to Matplotlib's Ellipse patch constructor.

        """
        self.ID = ID
        self.mode = mode
        self.a_des = 0
        self.a0 = 0
        self.ttc = None
        self.f_a = None
        self.alphas = None
        self.image = None
        self.leader = None
        self.rad = None
        self.gap = None
        self.tau = None
        self.lam = None
        self.v0 = None
        self.d = None
        self.pos = array((x, y))
        self.vel = array((vx, vy))
        self.theta = atan2(vy, vx)
        self.interactions = []
        if self.mode == "Car":
            self.l = 2.2  # half length
            self.w = 0.9  # half width
            self.a = 0.15326290995077607  # noise amplitude
            self.b = 3.379556325218917  # relaxation time
            self.p = 8.0
            self.q = 0.5
        else:
            self.l = 0.8  # half length
            self.w = 0.3  # half width
            self.a = 0.24670423672446218  # noise amplitude
            self.b = 1.30430934999647  # relaxation time
            self.p = 2.0
            self.q = 2.0
        self.styles = styles
        if not self.styles:
            # Default ellipse styles
            self.styles = {"ec": "k", "fill": False}

    # For convenience, map the components of the particle's position and
    # velocity vector onto the attributes x, y, vx and vy.
    @property
    def x(self):
        return self.pos[0]

    @x.setter
    def x(self, value):
        self.pos[0] = value

    @property
    def y(self):
        return self.pos[1]

    @y.setter
    def y(self, value):
        self.pos[1] = value

    @property
    def vx(self):
        return self.vel[0]

    @vx.setter
    def vx(self, value):
        self.vel[0] = value
        self.theta = atan2(self.vy, value)

    @property
    def vy(self):
        return self.vel[1]

    @vy.setter
    def vy(self, value):
        self.vel[1] = value
        self.theta = atan2(value, self.vx)

    def draw(self, ax):
        """Add this Particle's Ellipse patch to the Matplotlib Axes ax."""
        ellipse = Ellipse(xy=self.pos, width=2 * self.l, height=2 * self.w, angle=degrees(self.theta), **self.styles)
        ax.add_patch(ellipse)
        return ellipse

    def advance(self, dt, new_V, new_theta):
        """Advance the particle's position according to its velocity."""
        self.vx = new_V * cos(new_theta)
        self.vy = new_V * sin(new_theta)
        # apply periodic boundary conditions
        self.pos = self.pos + self.vel * dt
        if self.pos[0] > params.L / 2:
            self.pos[0] -= params.L

    def encode(self):
        return self.__dict__

    def __deepcopy__(self, memodict={}):
        # https://stackoverflow.com/questions/24756712
        copy_object = Particle(self.x, self.y, self.vx, self.vy, self.mode, self.ID, self.styles)
        copy_object.interactions = self.interactions
        copy_object.leader = self.leader
        copy_object.alphas = self.alphas
        copy_object.a_des = self.a_des
        copy_object.ttc = self.ttc
        copy_object.f_a = self.f_a
        copy_object.gap = self.gap
        copy_object.tau = self.tau
        copy_object.lam = self.lam
        copy_object.v0 = self.v0
        copy_object.a0 = self.a0
        copy_object.d = self.d
        return copy_object

    def __getitem__(self, key):
        """Get the value of a specific attribute of the particle."""
        return getattr(self, key)
