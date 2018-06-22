import os

import cv2
import numpy as np
import pandas as pd

'''
This code is to take each of the individual image-coordinate pairs and bundle them together to give a single file.
'''

local_repo_path = os.getcwd()

test3 = {}


def append_data(filename):
    global test3
    os.chdir("Data/TestImages")
    img = cv2.imread("{}.png".format(filename))
    os.chdir(local_repo_path + '\Data\TestCoords')
    coords = pd.read_csv("{}.csv".format(filename), header=0)

    # Splitting the colour channels of the original image into separate arrays.
    b, g, r = cv2.split(img)

    intermediate_b = pd.DataFrame(data=b)
    test = pd.concat([np.transpose(coords), intermediate_b], axis=1, keys=['Coords', 'Img'], join='outer', sort=True)
    test = test.transpose()
    test3[filename] = test
    os.chdir(local_repo_path)


for name in os.listdir(local_repo_path + '\Data\TestCoords'):
    append_data(name.split('.')[0])

test4 = pd.concat(test3.values(), axis=0, keys=test3.keys(), sort=True)
test4 = test4.apply(lambda x: pd.Series(x.dropna().values), 1)
test4.columns = test4.columns[:len(test4.columns)]
test4 = test4.reindex_axis(test4.columns, 1)
test4 = test4.dropna(axis=1, how='all')
print(test4)

test4.to_csv(local_repo_path + '\Data\MergeDat\Ting.csv')
