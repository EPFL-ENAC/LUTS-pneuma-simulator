from numpy import ndarray, roll, unique, zeros

from pNeuma_simulator.shadowcasting.fov import FoV


def shadowcasting(matrix: ndarray, origin: tuple, grid: float, L: float, d_max: float):
    """
    Compute visible interactions on a grid using shadowcasting.

    This function rolls the matrix so that the origin is centered, adds temporary
    walls to account for the horizon distance, computes the field of view using FoV,
    masks all non-visible cells, and returns a list of unique interacting IDs.

    Args:
        matrix (ndarray): The input grid of agents.
        origin (tuple): The (x, y) coordinates of the observer in grid units.
        grid (float): The grid resolution in meters.
        L (float): Total road length in meters.
        d_max (float): Horizon distance for visibility.

    Returns:
        ndarray: Array of unique interaction IDs.
    """
    height, width = matrix.shape

    # Center origin horizontally
    shift = int(width / 2 - origin[1])
    origin = (int(width / 2), origin[0])
    df = roll(matrix, shift, axis=1)

    # Add temporary walls
    left_wall = int((L / 2 - d_max) / grid)
    right_wall = int(1 + (L / 2 + d_max) / grid)
    df[:, [left_wall, right_wall]] += 1

    # Compute field of view
    fov = FoV(df, origin)
    is_visible = fov.compute_fov()

    # Remove temporary walls
    df[:, [left_wall, right_wall]] -= 1
    df[[0, -1]] = 0

    # Mask all non-visible cells
    visible_mask = zeros((height, width), dtype=bool)
    if is_visible:
        xs, ys = zip(*is_visible)
        visible_mask[ys, xs] = True
    df[~visible_mask] = 0

    # Return unique interactions
    interactions = unique(df)[1:]
    return interactions
