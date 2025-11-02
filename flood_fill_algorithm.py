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

goal = (5,11)
distances = np.full(size, np.inf)
distances[goal[0]][goal[1]] = 0

frames = []

def floodfill(maze, goal):
    queue = deque([goal])

    while queue:
        x,y = queue.popleft()
        directions = [(1,0),(-1,0),(0,1),(0,-1)]
        frames.append(distances.copy())
        for dx, dy in directions:
            nx , ny = x+dx, y+dy
            if 0 < nx < len(maze) and 0 < ny < len(maze[0]):
                if distances[nx][ny] == float('inf') and maze[nx][ny] == 1:
                    distances[nx][ny] = distances[x][y] + 1
                    queue.append((nx, ny)) 


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
            if 0 < nx < len(maze_final) and 0 < ny < len(maze_final[0]):
                if maze_final[nx][ny] == 1 and distances[nx][ny] < best_distance:
                    best_distance = distances[nx][ny]
                    best = (nx, ny)
        if best is None:
            return None
        path.append(current)
        current = best
    path.append(goal)
    return path

floodfill(maze=maze_final, goal=goal)
path = get_path(start=(1,1), goal=goal)

print(distances)


#Animation de la solution

fig, ax1 = plt.subplots(figsize = (12, 6))

ax1.imshow(maze_final, cmap="gray")
ax1.set_title("labyrinthe réduite")
ax1.axis('off')

max_dist = np.where(distances == np.inf, 0, distances)
max_dist = max_dist.max()
vmax = max(max_dist, 1)

ax1.plot(1,1,'go', markersize=8, label="Départ")
ax1.plot(goal[1], goal[0],'ro', markersize=8, label="Arrivée")
ax1.legend()

robots, = ax1.plot([], [], 'bo', markersize=8)

trace_robot, = ax1.plot([], [], '-', linewidth=3)

path_array = np.array(path)
Xs = []
Ys = []

def update(frame_idx):
    if frame_idx < len(frames):
        x, y = path_array[frame_idx]
        robots.set_data([y],[x])

        Xs.append(x)
        Ys.append(y)
        trace_robot.set_data(Ys, Xs)
    return  [robots]

ani = FuncAnimation(fig, update, frames=len(frames), interval = 50, repeat = False)

plt.tight_layout()
plt.show()
