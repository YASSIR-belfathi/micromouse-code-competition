import cv2
import matplotlib.pyplot as plt
import numpy as np


img  = cv2.imread("16 by 16 orthogonal maze(1).png", cv2.IMREAD_GRAYSCALE)

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)


maze = (binary/255).astype(int)

plt.imshow(maze, cmap='gray')
plt.show()

np.savetxt("value.csv", maze, fmt="%d", delimiter=",")

