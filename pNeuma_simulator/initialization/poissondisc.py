from copy import deepcopy

import numpy as np

from pNeuma_simulator import params
from pNeuma_simulator.contact_distance import ellipses
from pNeuma_simulator.gang.particle import Particle


class PoissonDisc:
    # Adapted from https://scipython.com/blog/poisson-disc-sampling-in-python
    def __init__(
        self,
        n_cars: int,
        n_moto: int,
        cell: float = 2.25,
        L: int = 180,
        W: float = 7.2,
        k: int = 30,
        aspect: float = 2.5,
        clearance: float = 0.25,
        rng=None,
    ):
        self.L, self.W = L, W
        self.n_cars = n_cars
        self.n_moto = n_moto
        # Cell side length
        self.cell = cell
        # Number of cells in the x- and y-directions of the grid
        self.nx, self.ny = int(L / self.cell) + 2, int(W / self.cell)
        self.width = (self.nx - 2) * self.cell
        self.height = self.ny * self.cell
        self.x_boundaries = np.arange(-self.nx * self.cell / 2, (self.nx + 1) * self.cell / 2, self.cell)
        self.y_boundaries = np.arange(-self.ny * self.cell / 2, (self.ny + 1) * self.cell / 2, self.cell)
        # Choose up to k points around each reference point as candidates for a new sample point
        self.k = k
        # Anisotropy parameter
        self.aspect = aspect
        # Minimum gap
        self.clearance = clearance
        # Pseudo-number generator
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        self.reset()

    def reset(self):
        """Reset the cells dictionary."""

        # A list of coordinates in the grid of cells
        coords_list = [(ix, iy) for ix in range(self.nx) for iy in range(self.ny)]
        # Initialize the dictionary of cells:
        # the grid must take a list of points.
        # http://devmag.org.za/2009/05/03/poisson-disk-sampling/
        self.cells = {coords: [] for coords in coords_list}

    def get_cell_coords(self, pt):
        """Get the coordinates of the cell that pt = (x,y) falls in."""
        # https://stackoverflow.com/questions/51333744
        x = np.digitize(pt.x, self.x_boundaries) - 1
        y = np.digitize(pt.y, self.y_boundaries) - 1
        return x, y

    def get_neighbours(self, coords):
        """Return the indexes of points in cells neighbouring cell at coords.

        For the cell at coords = (x,y), return the indexes of points in the cells
        with neighbouring coordinates illustrated below: ie those cells that could
        contain points closer than r.

                                        ooooo
                                        ooXoo
                                        ooooo

        """

        dxdy = [
            (-2, -1),
            (-1, -1),
            (0, -1),
            (1, -1),
            (2, -1),
            (-2, 0),
            (-1, 0),
            (0, 0),
            (1, 0),
            (2, 0),
            (-2, 1),
            (-1, 1),
            (0, 1),
            (1, 1),
            (2, 1),
        ]
        neighbours = []
        for dx, dy in dxdy:
            neighbour_coords = coords[0] + dx, coords[1] + dy
            if not (0 <= neighbour_coords[0] < self.nx and 0 <= neighbour_coords[1] < self.ny):
                # We're off the grid: no neighbours here.
                continue
            neighbour_cells = self.cells[neighbour_coords]
            # The grid must take a list of points.
            # http://devmag.org.za/2009/05/03/poisson-disk-sampling/
            for neighbour_cell in neighbour_cells:
                if neighbour_cell is not None:
                    # This cell is occupied: store this index of the contained point.
                    neighbours.append(neighbour_cell)
        return neighbours

    def point_valid(self, pt):
        """Is pt a valid point to emit as a sample?

        It must be no closer than r from any other point: check the cells in its
        immediate neighbourhood.

        """
        cell_coords = self.get_cell_coords(pt)
        for idx in self.get_neighbours(cell_coords):
            nearby_pt = self.samples[idx]
            # Manage periodic boundary conditions
            if nearby_pt.image:
                image_pt = nearby_pt.image
                if image_pt.x >= self.width / 2 and cell_coords[0] > (self.nx - 4):
                    d = ellipses(
                        pt.l,
                        pt.w,
                        image_pt.l,
                        image_pt.w,
                        pt.x,
                        pt.y,
                        image_pt.x,
                        image_pt.y,
                        0,
                        0,
                    )
                    # Squared distance between our candidate point, pt, and this image_pt.
                    distance2 = (image_pt.x - pt.x) ** 2 + (image_pt.y - pt.y) ** 2
                elif image_pt.x < self.width / 2 and cell_coords[0] == 1:
                    d = ellipses(
                        pt.l,
                        pt.w,
                        image_pt.l,
                        image_pt.w,
                        pt.x,
                        pt.y,
                        image_pt.x,
                        image_pt.y,
                        0,
                        0,
                    )
                    # Squared distance between our candidate point, pt, and this image_pt.
                    distance2 = (image_pt.x - pt.x) ** 2 + (image_pt.y - pt.y) ** 2
                else:
                    d = ellipses(
                        pt.l,
                        pt.w,
                        nearby_pt.l,
                        nearby_pt.w,
                        pt.x,
                        pt.y,
                        nearby_pt.x,
                        nearby_pt.y,
                        0,
                        0,
                    )
                    # Squared distance between our candidate point, pt, and this nearby_pt.
                    distance2 = (nearby_pt.x - pt.x) ** 2 + (nearby_pt.y - pt.y) ** 2
            else:
                d = ellipses(
                    pt.l,
                    pt.w,
                    nearby_pt.l,
                    nearby_pt.w,
                    pt.x,
                    pt.y,
                    nearby_pt.x,
                    nearby_pt.y,
                    0,
                    0,
                )
                # Squared distance between our candidate point, pt, and this nearby_pt.
                distance2 = (nearby_pt.x - pt.x) ** 2 + (nearby_pt.y - pt.y) ** 2
            if distance2 < (d + self.clearance) ** 2:
                # The points are too close, so pt is not a candidate.
                return False
        # All points tested: if we're here, pt is valid
        return True

    def get_point(self, ref_pt):
        """Try to find a candidate point relative to ref_pt to emit in the sample.

        We draw up to k points from the annulus of inner radius r, outer radius 2r
        around the reference point, ref_pt. If none of them are suitable (because
        they're too close to existing points in the sample), return False.
        Otherwise, return the pt.

        """
        # Minimum distance between samples
        if ref_pt.mode == "Car":
            r = 7 * self.cell / 5
        else:
            r = 4 * self.cell / 5
        i = 0
        while i < self.k:
            # https://meyavuz.wordpress.com/2018/11/15/
            rho = np.sqrt(self.rng.uniform(r**2, 4 * r**2))
            theta = self.rng.uniform(0, 2 * np.pi)
            x = ref_pt.x + rho * np.cos(theta)
            y = ref_pt.y + rho / self.aspect * np.sin(theta)
            pt = Particle(x, y, 0.0, 0.0, "Moto")
            if not (
                -self.width / 2 < pt.x < self.width / 2
                and -self.height / 2 + (pt.w + self.clearance) < pt.y < self.height / 2 - (pt.w + self.clearance)
            ):
                # This point falls outside the domain, so try again.
                continue
            if self.point_valid(pt):
                return pt
            i += 1
        # We failed to find a suitable point in the vicinity of refpt.
        return False

    def sample(self, rng):
        """Poisson disc random sampling in 2D.

        Draw random samples on the domain width x height such that no two
        samples are closer than r apart. The parameter k determines the
        maximum number of candidate points to be chosen around each reference
        point before removing it from the "active" list.

        """

        self.samples = []
        self.images = []
        active = []
        # Generate car instances in lanes.
        X = np.arange(start=-self.width / 2, stop=self.width / 2, step=self.width / self.n_cars)
        for x in X:
            for y in [-params.lane / 2, params.lane / 2]:
                ID = len(self.samples)
                pt = Particle(x, y, 0.0, 0.0, "Car", ID)
                self.samples.append(pt)
                cell_coords = self.get_cell_coords(pt)
                self.cells[cell_coords] = [ID]
                active.append(ID)
                if cell_coords[0] == 1:
                    image = deepcopy(pt)
                    image.x += self.width
                    image_coords = self.get_cell_coords(image)
                    self.cells[image_coords] = [ID]
                    image.styles["ls"] = "--"
                    self.images.append(image)
                    pt.image = image
        if self.n_moto > 0:
            while active:
                # choose a random "reference" point from the active list.
                idx = rng.choice(active)
                refpt = self.samples[idx]
                # Try to pick a new point relative to the reference point.
                pt = self.get_point(refpt)
                if pt:
                    # Point pt is valid: add it to the samples list and mark it as active
                    ID = len(self.samples)
                    pt.ID = ID
                    self.samples.append(pt)
                    cell_coords = self.get_cell_coords(pt)
                    self.cells[cell_coords].append(ID)
                    active.append(ID)
                    if cell_coords[0] == 1:
                        image = deepcopy(pt)
                        image.x += self.width
                        image_coords = self.get_cell_coords(image)
                        self.cells[image_coords].append(ID)
                        image.styles["ls"] = "--"
                        self.images.append(image)
                        pt.image = image
                    if cell_coords[0] == self.nx - 2:
                        image = deepcopy(pt)
                        image.x -= self.width
                        image_coords = self.get_cell_coords(image)
                        self.cells[image_coords].append(ID)
                        image.styles["ls"] = "--"
                        self.images.append(image)
                        pt.image = image
                else:
                    # We had to give up looking for valid points near refpt,
                    # so remove it from the list of "active" points.
                    active.remove(idx)

        return self.samples, self.images
