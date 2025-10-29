import numpy as np
import matplotlib.pyplot as plt
import random


def create_maze(width = 1, height = 1):
    array = np.ones((height*2+1, width*2+1), dtype='int')

    start_x, start_y=1,1
    array[start_x, start_y] = 0
    stack = [(start_x, start_y)]

    while stack:
        x,y = stack[-1]
        directions = [(2,0), (-2,0), (0,2), (0,-2)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x+dx, y+dy
            if 1 <= nx < height*2 and 1 <= ny < width*2 and array[nx, ny] == 1:
                array[x + dx//2, y + dy//2] = 0
                array[nx, ny] = 0
                stack.append((nx, ny))
                break
        else : 
            stack.pop()
    return array

array = create_maze(width=16, height=16)

print(array)

plt.imshow(array, cmap='gray')
plt.show()

