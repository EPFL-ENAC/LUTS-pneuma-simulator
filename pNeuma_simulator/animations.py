from copy import deepcopy
from math import cos, sin

import matplotlib.axes as mpl_axes
from matplotlib.patches import Ellipse
from numpy import degrees

from pNeuma_simulator import params


def draw(agent: dict, n: int, n_cars: int, ax: mpl_axes.Axes) -> None:
    """Add this serialized agent's Ellipse patch to the Matplotlib Axes ax.

    Args:
        agent (dict): The agent object to be drawn.
        n (int): positional index of agent
        n_cars (int): The number of cars in the dataset.
        ax (mpl_axes.Axes): The Matplotlib Axes object to add the patch to.
    """
    theta = agent["theta"]
    speed = agent["speed"]
    if n <= 2 * n_cars - 1:
        width = 2 * params.car_l
        height = 2 * params.car_w
    else:
        width = 2 * params.moto_l
        height = 2 * params.moto_w
    ellipse = Ellipse(
        xy=agent["pos"],
        width=width,
        height=height,
        angle=degrees(theta),
        **{"ec": "k", "fc": "w", "lw": 0.5},
        clip_on=True
    )
    ax.add_patch(ellipse)
    ax.scatter(
        agent["pos"][0],
        agent["pos"][1],
        marker="o",
        fc="k",
        ec="none",
        s=4,
    )
    if speed > 0:
        ax.arrow(
            agent["pos"][0],
            agent["pos"][1],
            speed * cos(theta),
            speed * sin(theta),
            antialiased=True,
            lw=0.25,
            width=0.1,
            head_width=0.5,
            head_length=0.75,
            color="k",
            zorder=2,
        )


def ring(t: int, n_cars: int, l_agents: list, ax: mpl_axes.Axes) -> None:
    """
    Draw the ring animation for a given time step.

    Args:
        t (int): The time step.
        n_cars (int): The number of cars in the dataset.
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
        lw=1,
    )
    for n, agent in enumerate(l_agents[t]):
        if agent["pos"][0] < -params.d_max:
            image = deepcopy(agent)
            image["pos"][0] += params.L
            draw(image, n, n_cars, ax)
        elif agent["pos"][0] > params.d_max:
            image = deepcopy(agent)
            image["pos"][0] -= params.L
            draw(image, n, n_cars, ax)
        draw(agent, n, n_cars, ax)
