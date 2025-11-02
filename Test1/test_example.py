import numpy as np
import matplotlib.pyplot as plt

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
            if 0 <= nx < h and 0 <= ny < w:
                if maze[nx][ny] == 0:
                    if distances[nx, ny]>distances[x,y]+1:
                        distances[nx, ny] = distances[x,y]+1
                        stack.append((nx,ny))

    return distances


maze = [[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]]

distances = flood_fill_algorithm(maze,(3,3))

print(distances)

start_point = (1,1)

path = [start_point]

x,y = start_point

path_founded : bool = True

while distances[x,y] != 0:
    min_values = distances[x,y]
    next_pos = None

    for dx,dy in [(1,0), (-1,0), (0, 1), (0, -1)]:
        nx, ny= x+dx, y+dy
        if 0 <= nx < distances.shape[0] and 0 <= ny < distances.shape[0]:
            if distances[nx, ny] < min_values:
                min_values = distances[nx, ny]
                next_pos = (nx, ny)
    if next_pos is None :
        print("there is no path")
        path_founded = False
        break
    else:
        x,y = next_pos
        path.append(next_pos)

if path_founded:
    print(f"the path is {path}")


for index_line in range(len(maze[0])):
    for index_column in range(len(maze[index_line])):
        if maze[index_line][index_column] == 0:
            maze[index_line][index_column] = 1
        else:
            maze[index_line][index_column] = 0

plt.imshow(maze, cmap="gray")
plt.show()
