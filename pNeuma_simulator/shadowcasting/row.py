from math import ceil, floor


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
