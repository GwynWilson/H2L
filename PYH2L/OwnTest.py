import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import RectangleSelector

"""
This script allows the reading of the test images as arrays, splits the colour
channels, displays one of the channels using imshow. A draggable box can then
be drawn and the coordinates of two opposite corners of the rectangle (not sure
which ones) are printed.
"""


def func(filename):
    # Test images are now within repository so this line will set the correct directory as long as your local version is
    # up to date
    local_repo_path = os.getcwd()
    os.chdir("Data/TestImages")
    # Reading the image as an array.
    img = cv2.imread("{}.png".format(filename))

    # Splitting the colour channels of the original image into separate arrays.
    b, g, r = cv2.split(img)

    # Using fig, ax to make the interactive bit work.
    fig, ax = plt.subplots()
    plt.imshow(r)

    coords = pd.DataFrame(columns=['blx', 'bly', 'trx', 'try'])

    ix = 0

    def line_select_callback(eclick, erelease, ix=ix):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        rect = plt.Rectangle((min(x1, x2), min(y1, y2)), np.abs(x1 - x2), np.abs(y1 - y2))
        coords.at[ix, 'blx'] = int(x1)
        coords.at[ix, 'bly'] = int(y1)
        coords.at[ix, 'trx'] = int(x2)
        coords.at[ix, 'try'] = int(y2)
        ix += 1
        ax.add_patch(rect)

    rs = RectangleSelector(ax, line_select_callback,
                           drawtype='box', useblit=False, button=[1],
                           minspanx=5, minspany=5, spancoords='pixels',
                           interactive=True)

    plt.show()
    os.chdir(local_repo_path + '\Data\TestCoords')
    coords.to_csv('{}.csv'.format(filename))


func('Layer 2')
