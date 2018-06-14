import os

import cv2
import numpy as np
import pandas as pd

'''
This code is to take each of the individual image-coordinate pairs and bundle them together to give a single file.
'''

local_repo_path = os.getcwd()

test2 = pd.DataFrame(columns=['Identifier', 'Data'])


def append_data(filename):
    os.chdir("Data/TestImages")
    img = cv2.imread("{}.png".format(filename))
    os.chdir(local_repo_path + '\Data\TestCoords')
    coords = pd.read_csv("{}.csv".format(filename), header=0)

    # Splitting the colour channels of the original image into separate arrays.
    b, g, r = cv2.split(img)

    print(b.shape)
    print(coords)
    intermediate_b = pd.DataFrame(data=b)
    test = pd.concat([np.transpose(coords), intermediate_b], axis=1, keys=['Coords', 'Img'], join='outer', sort=True)
    print(test)
    test = test.transpose()
    print(test)
    print(test.shape)
    test2.append({'Identifier': filename, 'Data': test})


for name in os.listdir(local_repo_path + '\Data\TestImages'):
    print(local_repo_path + '\Data\TestImages')
    print(name)
    append_data(name.split('.')[0])
print(test2)
