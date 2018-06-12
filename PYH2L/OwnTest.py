import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector

"""
This script allows the reading of the test images as arrays, splits the colour
channels, displays one of the channels using imshow. A draggable box can then
be drawn and the coordinates of two opposite corners of the rectangle (not sure
which ones) are printed.
"""

# TODO: Create function to write the coordinates as a .csv file with a name linking it to its associated image file.

os.chdir("C:\\Users\Josh\Desktop\TestSet")

img = cv2.imread("Layer 2.png")
b, g, r = cv2.split(img)

fig, ax = plt.subplots()
plt.imshow(r)


def line_select_callback(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    rect = plt.Rectangle((min(x1, x2), min(y1, y2)), np.abs(x1 - x2), np.abs(y1 - y2))
    BoxCoords = [[int(x1), int(y1)], [int(x2), int(y2)]]
    print(BoxCoords)
    ax.add_patch(rect)


rs = RectangleSelector(ax, line_select_callback,
                       drawtype='box', useblit=False, button=[1],
                       minspanx=5, minspany=5, spancoords='pixels',
                       interactive=True)

plt.show()
