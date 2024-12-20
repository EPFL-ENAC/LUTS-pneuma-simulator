from pNeuma_simulator.gang.particle import Particle


def neighborhood(ego: Particle, candidates: list[Particle]) -> list[Particle]:
    """Constructs a neighborhood with periodic boundaries.

    Args:
        ego (Particle): ego vehicle.
        candidates (list): candidate vehicles.

    Returns:
        list: neighboring vehicles.
    """
    neighbors = []
    interactions = ego.interactions
    for interaction in interactions:
        other = candidates[int(interaction - 1)]
        if other.image:
            if abs(ego.x - other.x) > abs(ego.x - other.image.x):
                neighbors.append(other.image)
            else:
                neighbors.append(other)
        else:
            neighbors.append(other)
    return neighbors
