from math import atan2, cos, degrees, sin

from matplotlib.patches import Ellipse
from numpy import array

from pNeuma_simulator import params


class Particle:
    """A class representing a two-dimensional particle.

    This class models a particle with position, velocity, and other attributes
    relevant to its motion and interactions.

    Attributes:
        ID (int): The unique identifier of the particle.
        mode (str): Type of particle, e.g., "Car" or "Moto".
        a0 (float): The initial acceleration.
        ttc (float): Time to collision.
        f_a (list): Distance to collision.
        image (object): Clone of the particle.
        leader (Particle): The leading particle.
        rad (numpy.ndarray): The radius matrix.
        gap (float): The gap between particles.
        tau (float): Adaptation time.
        lam (float): Lambda parameter.
        v0 (float): Desired velocity.
        s0 (float): Jam spacing.
        pos (numpy.ndarray): The position of the particle.
        vel (numpy.ndarray): The velocity of the particle.
        theta (float): The angle of the particle's velocity.
        interactions (list): List of interactions with other particles.
    """

    def __init__(self, x, y, vx, vy, mode="Car", ID=None, styles=None):
        """Initialize the particle's position, velocity, mode and ID.

        Any key-value pairs passed in the styles dictionary will be passed
        as arguments to Matplotlib's Ellipse patch constructor.

        Args:
            x (float): The x-coordinate of the particle's position.
            y (float): The y-coordinate of the particle's position.
            vx (float): The x-component of the particle's velocity.
            vy (float): The y-component of the particle's velocity.
            mode (str, optional): The mode of the particle. Defaults to "Car".
            ID (int, optional): The unique identifier of the particle. Defaults to None.
            styles (dict, optional): A dictionary of styles for Matplotlib's Ellipse patch. Defaults to None.
        """
        self.ID = ID
        self.mode = mode
        self.a0 = 0
        self.ttc = None
        self.f_a = None
        self.image = None
        self.leader = None
        self.rad = None
        self.gap = None
        self.tau = None
        self.lam = None
        self.v0 = None
        self.s0 = None
        self.pos = array((x, y))
        self.vel = array((vx, vy))
        self.theta = atan2(vy, vx)
        self.interactions = []
        if self.mode == "Car":
            self.l = params.car_l  # half length
            self.w = params.car_w  # half width
            self.a = params.car_a  # noise amplitude
            self.b = params.car_b  # relaxation time
        else:
            self.l = params.moto_l  # half length
            self.w = params.moto_w  # half width
            self.a = params.moto_a  # noise amplitude
            self.b = params.moto_b  # relaxation time
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

    def encode(self, t):
        my_dict = self.__dict__
        my_dict.pop("ID", None)
        my_dict.pop("mode", None)
        my_dict.pop("image", None)
        my_dict.pop("leader", None)
        my_dict.pop("gap", None)
        my_dict.pop("rad", None)
        my_dict.pop("f_a", None)
        my_dict.pop("a0", None)
        my_dict.pop("l", None)
        my_dict.pop("w", None)
        my_dict.pop("a", None)
        my_dict.pop("b", None)
        my_dict.pop("styles", None)
        my_dict.pop("interactions", None)
        if t > 0:
            my_dict.pop("tau", None)
            my_dict.pop("lam", None)
            my_dict.pop("v0", None)
            my_dict.pop("s0", None)
        return my_dict

    def __deepcopy__(self, memodict={}):
        # https://stackoverflow.com/questions/24756712
        copy_object = Particle(self.x, self.y, self.vx, self.vy, self.mode, self.ID, self.styles)
        copy_object.interactions = self.interactions
        copy_object.leader = self.leader
        copy_object.ttc = self.ttc
        copy_object.f_a = self.f_a
        copy_object.gap = self.gap
        copy_object.tau = self.tau
        copy_object.lam = self.lam
        copy_object.v0 = self.v0
        copy_object.a0 = self.a0
        copy_object.s0 = self.s0
        return copy_object

    def __getitem__(self, key):
        """Get the value of a specific attribute of the particle."""
        return getattr(self, key)
