import cv2
import matplotlib.pyplot as plt
import numpy as np


img  = cv2.imread("WhatsApp Image 2025-10-24 at 17.55.22.jpeg", cv2.IMREAD_GRAYSCALE)

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

maze = (binary/255).astype(int)

# print(maze)

# plt.imshow(maze, cmap="gray")
# plt.show()

# print(img.shape)

height, width = maze.shape
n_cells = 16
h_cells = height // n_cells
w_cells = width // n_cells

new_maze = np.zeros((n_cells, n_cells), dtype=int)

for i in range(n_cells):
    for j in range(n_cells):
        cell = binary[i*h_cells:(i+1)*h_cells, j*w_cells:(j+1)*w_cells]
        if np.mean(cell) > 127 :
            new_maze[i,j] = 0
        else:
            new_maze[i, j] = 1

print(new_maze)

plt.imshow(new_maze, cmap="gray")
plt.show()

# print(img.shape)

