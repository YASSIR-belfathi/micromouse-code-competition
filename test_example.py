import numpy as np

def Shape(maze):
    row = len(maze)
    if row > 0:
        column = len(maze[0])
    return (row, column)

def flood_fill_algorithm(maze , goal):
    h,w = Shape(maze)
    distances = np.full((h,w), np.inf)
    distances[goal] = 0

    stack = [goal]

    while stack:
        x, y = stack.pop()
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 < nx <h and 0<y<w:
                if maze[nx][ny] == 0:
                    if distances[nx, ny]>distances[x,y]+1:
                        distances[nx, ny] = distances[x,y]+1
                        stack.append((nx,ny))

    return distances


maze = [[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]]

distances = flood_fill_algorithm(maze,(3,3))

print(distances)