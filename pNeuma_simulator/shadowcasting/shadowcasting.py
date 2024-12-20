import numpy as np

from pNeuma_simulator.shadowcasting.fov import FoV


def shadowcasting(matrix: np.ndarray, origin: tuple, grid: float, L: float, d_max: float) -> list:
    """https://www.albertford.com/shadowcasting/

    Args:
        matrix (np.ndarray): background grid
        origin (tuple): position on the grid
        grid (float): grid size in meters
        L (float): road length in meters
        d_max (float): horizon distance

    Returns:
        list: list of interactions
    """
    height, width = matrix.shape
    roll = int(width / 2 - origin[1])
    origin = (origin[1] + roll, origin[0])
    df = np.roll(matrix, roll, axis=1)
    # Add temporary walls
    df[:, [int(-1 + (L / 2) / grid), int(1 + (L / 2 + d_max) / grid)]] += 1
    fov = FoV(df, origin)
    is_visible = fov.compute_fov()
    # Remove all the walls
    df[:, [int(-1 + (L / 2) / grid), int(1 + (L / 2 + d_max) / grid)]] -= 1
    df[[0, -1]] = 0
    for y in range(height):
        for x in range(width):
            if (x, y) not in is_visible:
                df[y, x] = 0
    interactions = np.unique(df)[1:]
    interactions = interactions[interactions > 0]
    return interactions
