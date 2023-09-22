from collections import deque

# Another adapted solution from the Internet
# These arrays are used to get row and
# column numbers of 8 neighbours of a given cell
ROW_DIRECT = [-1, 0, 0, 1]
COL_DIRECT = [0, -1, 1, 0]


# A function to check if a given cell
# (u, v) can be included in DFS
def is_safe(mat, x, y, processed):
    in_row = x >= 0 and x < len(processed)
    in_col = y >= 0 and y < len(processed[0])
              
    return in_row and in_col and mat[x][y] == 1 and not processed[x][y]
 

def BFS(mat, processed, i, j):
    # Simple BFS first step, we enqueue source and mark it as visited
    q = deque()
    q.append((i, j))
    processed[i][j] = True
 
    # Next step of BFS. We take out items one by one from queue and
    # enqueue their unvisited adjacent
    while q:
        x, y = q.popleft()
        # Go through all 8 adjacent
        for k in range(len(ROW_DIRECT)):

            if is_safe(mat, x + ROW_DIRECT[k], y + COL_DIRECT[k], processed):

                processed[x + ROW_DIRECT[k]][y + COL_DIRECT[k]] = True
                q.append((x + ROW_DIRECT[k], y + COL_DIRECT[k]))

                
# This function returns number islands (connected
# components) in a graph. It simply works as
# BFS for disconnected graph and returns count of BFS calls.
def count_islands(mat):
    if not mat or not len(mat):
        return 0
    (M, N) = (len(mat), len(mat[0]))
 
    # Mark all cells as not visited
    processed = [[False for x in range(N)] for y in range(M)]
 
    island = 0
    for i in range(M):
        for j in range(N):
            if mat[i][j] == 1 and not processed[i][j]:
                BFS(mat, processed, i, j)
                island = island + 1
 
    return island


if __name__ == '__main__':

    mat = [[1, 1, 0, 0, 0],
           [0, 1, 0, 0, 1],
           [1, 0, 0, 1, 1],
           [0, 0, 0, 0, 0],
           [1, 0, 1, 0, 1]]
 
    print('The total number of islands is', count_islands(mat))
