from fractions import Fraction
from math import ceil, floor


class FoV:
    def __init__(self, df, origin):
        self.is_visible = set()
        self.map_str = "\n".join("".join("#" if i > 0 else "." for i in j) for j in df)
        self.map_list = list(self.map_str.splitlines())
        self.origin = origin

    def is_blocking(self, x, y):
        return self.map_list[y][x] == "#"

    def mark_visible(self, x, y):
        self.is_visible.add((x, y))

    def compute_fov(self):
        self.mark_visible(*self.origin)

        for i in range(4):
            quadrant = Quadrant(i, self.origin)

            def reveal(tile):
                x, y = quadrant.transform(tile)
                self.mark_visible(x, y)

            def is_wall(tile):
                if tile is None:
                    return False
                x, y = quadrant.transform(tile)
                return self.is_blocking(x, y)

            def is_floor(tile):
                if tile is None:
                    return False
                x, y = quadrant.transform(tile)
                return not self.is_blocking(x, y)

            def slope(tile):
                row_depth, col = tile
                return Fraction(2 * col - 1, 2 * row_depth)

            def is_symmetric(row, tile):
                row_depth, col = tile
                return col >= row_depth * row.start_slope and col <= row_depth * row.end_slope

            def scan(row):
                prev_tile = None
                for tile in row.tiles():
                    if is_wall(tile) or is_symmetric(row, tile):
                        reveal(tile)
                    if is_wall(prev_tile) and is_floor(tile):
                        row.start_slope = slope(tile)
                    if is_floor(prev_tile) and is_wall(tile):
                        next_row = row.next()
                        next_row.end_slope = slope(tile)
                        scan(next_row)
                    prev_tile = tile
                if is_floor(prev_tile):
                    scan(row.next())

            first_row = Row(1, Fraction(-1), Fraction(1))
            scan(first_row)
        return self.is_visible


class Quadrant:
    north = 0
    east = 1
    south = 2
    west = 3

    def __init__(self, cardinal, origin):
        self.cardinal = cardinal
        self.ox, self.oy = origin

    def transform(self, tile):
        row, col = tile
        if self.cardinal == self.north:
            return (self.ox + col, self.oy - row)
        if self.cardinal == self.south:
            return (self.ox + col, self.oy + row)
        if self.cardinal == self.east:
            return (self.ox + row, self.oy + col)
        if self.cardinal == self.west:
            return (self.ox - row, self.oy + col)


class Row:
    def __init__(self, depth, start_slope, end_slope):
        self.depth = depth
        self.start_slope = start_slope
        self.end_slope = end_slope

    def round_ties_up(self, n):
        return floor(n + 0.5)

    def round_ties_down(self, n):
        return ceil(n - 0.5)

    def tiles(self):
        min_col = self.round_ties_up(self.depth * self.start_slope)
        max_col = self.round_ties_down(self.depth * self.end_slope)
        for col in range(min_col, max_col + 1):
            yield (self.depth, col)

    def next(self):
        return Row(self.depth + 1, self.start_slope, self.end_slope)
