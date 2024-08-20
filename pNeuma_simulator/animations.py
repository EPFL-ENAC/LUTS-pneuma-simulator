import matplotlib.axes as mpl_axes
from matplotlib.patches import Ellipse
from numpy import arange, degrees
from numpy.linalg import norm

from pNeuma_simulator import params
from pNeuma_simulator.gang.particle import Particle


def draw(particle: Particle, ax: mpl_axes.Axes) -> None:
    """
    Add this Particle's Ellipse patch to the Matplotlib Axes.

    Args:
        particle (Particle): The particle object to be drawn.
        ax (mpl_axes.Axes): The Matplotlib Axes object to add the patch to.
    """
    ellipse = Ellipse(
        xy=particle["pos"],
        width=2 * particle["l"],
        height=2 * particle["w"],
        angle=degrees(particle["theta"]),
        **particle["styles"]
    )
    ax.add_patch(ellipse)


def ring(t: int, l_agents: list, ax: mpl_axes.Axes, sampler) -> None:
    """
    Draw the ring animation for a given time step.

    Args:
        t (int): The time step.
        l_agents (list): A list of agents at each time step.
        ax (matplotlib.axes.Axes): The matplotlib axes object to draw the animation on.
        sampler: The sampler object.
    """

    for agent in l_agents[t]:
        draw(agent, ax)
        ax.scatter(agent["pos"][0], agent["pos"][1], marker="o", c="k", s=10, lw=1)
        if norm(agent["vel"]) > 0:
            ax.arrow(
                agent["pos"][0],
                agent["pos"][1],
                agent["vel"][0],
                agent["vel"][1],
                width=0.1,
                head_width=0.5,
                head_length=0.75,
                color=agent["styles"]["ec"],
                zorder=2,
            )
    ax.hlines(
        [-params.lane, 0, params.lane],
        xmin=-params.L / 2,
        xmax=params.L / 2,
        color="k",
        ls=["-", (0, (15, 15)), "-"],
        lw=[2, 1, 2],
    )
    ax.set_yticks(arange(-params.lane, 3 * params.lane / 2, params.lane / 2))
    ax.set_xlim(-(sampler.nx + 1) * params.cell / 2, (sampler.nx + 1) * params.cell / 2)
