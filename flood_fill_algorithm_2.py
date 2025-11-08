import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import cv2
from matplotlib.animation import FuncAnimation


#traitement d'image du labyrinthe
image = cv2.imread("WhatsApp Image 2025-10-24 at 17.55.22.jpeg", flags=cv2.IMREAD_GRAYSCALE)
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

size = maze_final.shape

goal = (15,7)
distances = np.full(size, np.inf)
distances[goal[0]][goal[1]] = 0

frames = []

#flood fill algorithm
def floodfill(maze, goal):
    queue = deque([goal])

    while queue:
        x,y = queue.popleft()
        directions = [(1,0),(-1,0),(0,1),(0,-1)]
        # frames.append(distances.copy())
        for dx, dy in directions:
            nx , ny = x+dx, y+dy
            if 0 < nx < len(maze) and 0 < ny < len(maze[0]):
                if distances[nx][ny] == float('inf') and maze[nx][ny] == 1:
                    distances[nx][ny] = distances[x][y] + 1
                    queue.append((nx, ny)) 

#get the path function
def get_path(start, goal):
    path = []
    current = start
    while current != goal:
        x,y = current
        best = None
        best_distance = float('inf')
        directions = [(1,0),(-1,0),(0,1),(0,-1)]
        for dx, dy in directions : 
            nx, ny = x+dx, y+dy
            if ((0 <= nx < len(maze_final)) and (0 < ny < len(maze_final[0])) 
                and maze_final[nx][ny] == 1 and distances[nx][ny] < best_distance):
                    best_distance = distances[nx][ny]
                    best = (nx, ny)
        if best is None:
            return None
        path.append(current)
        current = best
    path.append(goal)
    return path

floodfill(maze=maze_final, goal=goal)
path = get_path(start=(31,1), goal=goal)

print(distances)
