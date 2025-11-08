import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import heapq
import cv2
from matplotlib.animation import FuncAnimation
import time

#start_time
start_time = time.time()

#traitement d'image du labyrinthe
image = cv2.imread("WhatsApp Image 2025-11-06 at 19.34.58.jpeg", flags=cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
binary = cv2.bitwise_not(binary)

maze2 = (binary <= 127).astype(int)

#réduction de la taille de l'image sans perdre d'informations
reduced_rows = deque()
previous_row = None

for row in maze2:
    if previous_row is None or not np.array_equal(row, previous_row):
        reduced_rows.append(row)
        previous_row = row

maze_reduced_v = np.array(reduced_rows).T #.T is for transposition

cols_reduced = deque()
previous_col = None

for col in maze_reduced_v:
    if previous_col is None or not np.array_equal(col, previous_col):
        cols_reduced.append(col)
        previous_col = col

maze_final = np.array(cols_reduced).T

np.savetxt("maze1.txt", maze_final, delimiter=",", fmt="%d")

def heuristic(position, goal):
    return abs(position[0]-goal[0])+abs(position[1]-goal[1])

def astar_algorithm(maze, start, goal):
    rows, cols = maze.shape
    open_list = []
    heapq.heappush(open_list, (0+heuristic(start, goal), 0, start, [start]))
    visited = set()

    while open_list:
        f, g, current, path = heapq.heappop(open_list)
        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)

        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = current[0]+dx, current[1]+dy
            if (0 <= nx < rows) and (0 <= ny < cols) and (maze[nx,ny] == 1) and ((nx, ny) not in visited):
                new_cost = g + 1
                new_path = path + [(nx, ny)]
                heapq.heappush(open_list, (new_cost + heuristic((nx, ny), goal), new_cost, (nx,ny), new_path))
    return None

def find_dead_ends(maze):
    rows, cols = maze.shape
    dead_ends = []
    directions = [(-1,0),(1,0),(0,-1),(0,1)]  # haut, bas, gauche, droite

    for x in range(rows):
        for y in range(cols):
            if maze[x, y] == 1:  # seulement les chemins
                free_neighbors = 0
                for dx, dy in directions:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < rows and 0 <= ny < cols and maze[nx, ny] == 1:
                        free_neighbors += 1
                if free_neighbors == 1:
                    # Vérifie que c'est sur le bord (coin)
                    if ((x == 1 and y==1) or (x == rows-2 and y ==1) 
                        or(x==1 and y==cols-2) or(x==rows-2 and y==cols-2)):
                        dead_ends.append((x, y))
    return dead_ends

start = find_dead_ends(maze_final)[0]

path = astar_algorithm(maze=maze_final, start=start, goal=(51,15))

fig, ax = plt.subplots()
ax.imshow(maze_final, cmap='gray')

if path:
    px, py = zip(*path)
    ax.plot(py, px, color='red', linewidth=2, label='Chemin trouvé')[0]
    ax.scatter(start[1], start[0], color="green", s=50, label="départ")
    ax.scatter(15, 51, color="red", s=50, label="Arrivée")
    ax.legend()
else:
    print("path not found")

# ani = FuncAnimation(fig, animate, frames=len(path), interval=50, repeat=False)

#finish_time
total_time = time.time() - start_time
print(total_time)

plt.title("Résolution du labyrinthe avec A*")
plt.show()