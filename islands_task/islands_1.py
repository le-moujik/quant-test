from collections import deque


# Adapted solution from the Internet
class IslandCounter(object):
    def _append_if(self, queue, x, y):
        # Append to the queue only if in bounds 
        # of the grid and the cell value is 1.
        if 0 <= x < self.row_length and 0 <= y < self.col_length:
            if self.grid[x][y] == 1:
                queue.append((x, y))

    def _mark_neighbors(self, row, col):
        # Mark all the cells in the current island with value = 2. 
        # Breadth-first search.
        queue = deque()

        queue.append((row, col))
        while queue:
            x, y = queue.pop()
            self.grid[x][y] = 2

            self._append_if(queue, x - 1, y)
            self._append_if(queue, x, y - 1)
            self._append_if(queue, x + 1, y)
            self._append_if(queue, x, y + 1)

    def num_islands(self, grid):
        if not grid or len(grid) == 0 or len(grid[0]) == 0:
            return 0

        self.grid = grid
        self.row_length = len(grid)
        self.col_length = len(grid[0])

        island_counter = 0
        for row in range(self.row_length):
            for col in range(self.col_length):
                if self.grid[row][col] == 1:
                    # found an island
                    island_counter += 1
                    self._mark_neighbors(row, col)

        return island_counter


if __name__ == '__main__':
    grid = [[1, 1, 0, 0, 0],
            [0, 1, 0, 0, 1],
            [1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1]]

    print('The total number of islands is', IslandCounter().num_islands(grid))
