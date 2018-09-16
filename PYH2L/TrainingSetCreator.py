import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import RectangleSelector, Cursor

plt.switch_backend('QT5Agg')
positive_response = ['Yes', 'yes', 'y', 'yeah', 'Yeah', 'Y']

"""
This script allows the reading of the test images as arrays, splits the colour
channels, displays one of the channels using imshow. A draggable box can then
be drawn and the coordinates of two opposite corners of the rectangle (not sure
which ones) are printed.
"""

ix = -1

local_repo_path = os.getcwd()

initial_check = input("Overwrite any pre-existing coordinates?")


def func(filename):

    def func2(filename2=filename):
        os.chdir("Data/TestImages")
        # Reading the image as an array.
        img = cv2.imread("{}.png".format(filename2))
        # Using fig, ax to make the interactive bit work.
        fig, ax = plt.subplots()
        plt.imshow(img)

        coords = pd.DataFrame(columns=['blx', 'bly', 'trx', 'try'])

        def line_select_callback(eclick, erelease):
            global ix
            ix += 1
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata

            rect = plt.Rectangle((min(x1, x2), min(y1, y2)), np.abs(x1 - x2), np.abs(y1 - y2), alpha=0.3)
            coords.at[ix, 'blx'] = int(x1)
            coords.at[ix, 'bly'] = int(y1)
            coords.at[ix, 'trx'] = int(x2)
            coords.at[ix, 'try'] = int(y2)
            print(ix)
            ax.add_patch(rect)

        rs = RectangleSelector(ax, line_select_callback,
                               drawtype='box', useblit=False, button=[1],
                               minspanx=5, minspany=5, spancoords='pixels',
                               interactive=True)

        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()
        cursor = Cursor(ax, useblit=True, color='red', linewidth=1)
        plt.show()
        os.chdir(local_repo_path + '\Data\TestCoords')
        if not coords.empty:
            coords.to_csv('{}.csv'.format(filename2), index=False)

    if filename + '.csv' in os.listdir(local_repo_path + "\Data\TestCoords"):
        check = input("Coordinate data for this image ({}) is already present. Overwrite pre-existing coordinates?".format(filename))
        if check in positive_response:
            return func2()
        else:
            os.chdir(local_repo_path)
            pass
    else:
        os.chdir(local_repo_path)
        return func2()


# func('Layer 2')


files = [os.path.splitext(filename)[0] for filename in os.listdir(local_repo_path + "\Data\TestImages")]
for names in files:
    if names + '.csv' in os.listdir(local_repo_path + "\Data\TestCoords"):
        if initial_check not in positive_response:
            continue
    func(names)
