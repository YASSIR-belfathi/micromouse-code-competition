import cv2
import matplotlib.pyplot as plt
import numpy as np


img  = cv2.imread("WhatsApp Image 2025-10-24 at 17.55.22.jpeg", cv2.IMREAD_GRAYSCALE)

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

maze = (binary/255).astype(int)

print(maze)

plt.imshow(maze, cmap="gray")
plt.show()

print(img.shape)

array = np.array(maze)

np.savetxt("value.csv", array, delimiter = "," , fmt="%d" )



