import numpy as np


maze = [
    [0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,0,0],
    [0,0,1,0,1,0,1,0,0],
    [0,0,1,1,1,1,1,0,0],
    [0,0,1,0,1,0,1,0,0],
    [0,0,1,1,1,1,1,0,0],
    [0,0,0,0,0,0,0,0,0]
]

line = [1 for _ in range(5)]

for i in range(len(maze)-1):
    for j in range(len(maze[0])-1):
        center = maze[i:i+5, j:j+5]
        equals = 0
        not_equals = 0
        for rows in center :
            if np.array_equal(rows, line):
                equals += 1
            else:
                not_equals += 1
        if equals > not_equals:
            