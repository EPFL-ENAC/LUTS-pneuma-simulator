from copy import deepcopy
from numpy import degrees
from numpy.linalg import norm
from matplotlib.patches import Ellipse
import matplotlib.axes as mpl_axes

from pNeuma_simulator import params


def draw(agent: dict, ax: mpl_axes.Axes) -> None:
    """Add this serialized agent's Ellipse patch to the Matplotlib Axes ax.

    Args:
        agent (dict): The agent object to be drawn.
        ax (mpl_axes.Axes): The Matplotlib Axes object to add the patch to.
    """
    if agent["mode"] == "Car":
        width = 2 * params.car_l
        height = 2 * params.car_w
    else:
        width = 2 * params.moto_l
        height = 2 * params.moto_w
    ellipse = Ellipse(
        xy=agent["pos"],
        width=width,
        height=height,
        angle=degrees(agent["theta"]),
        **{"ec": "k", "fc": "w"},
    )
    ax.add_patch(ellipse)
    ax.scatter(
        agent["pos"][0],
        agent["pos"][1],
        marker="o",
        fc="k",
        ec="none",
        s=5,
    )
    if norm(agent["vel"]) > 0:
        ax.arrow(
            agent["pos"][0],
            agent["pos"][1],
            agent["vel"][0],
            agent["vel"][1],
            antialiased=True,
            width=0.1,
            head_width=0.5,
            head_length=0.75,
            color="k",
            zorder=2,
        )


def ring(t: int, l_agents: list, ax: mpl_axes.Axes) -> None:
    """
    Draw the ring animation for a given time step.

    Args:
        t (int): The time step.
        l_agents (list): A list of agents at each time step.
        ax (matplotlib.axes.Axes): The matplotlib axes object to draw the animation on.
    """
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-params.L / 2, params.L / 2)
    ax.set_ylim(-params.lane, params.lane)
    ax.set_axis_off()
    ax.hlines(
        [-params.lane, params.lane],
        xmin=-params.L / 2,
        xmax=params.L / 2,
        color="k",
        ls="-",
        lw=3,
    )
    for agent in l_agents[t]:
        if agent["pos"][0] < params.d_max:
            image = deepcopy(agent)
            image["pos"][0] += params.L
            draw(image, ax)
        elif agent["pos"][0] > params.d_max:
            image = deepcopy(agent)
            image["pos"][0] -= params.L
            draw(image, ax)
        draw(agent, ax)
