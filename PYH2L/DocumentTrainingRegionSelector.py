import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sc
from matplotlib.widgets import RectangleSelector, Cursor
import cv2
import os
import scipy.misc as sm
from tkinter import filedialog, Tk
import pdf2image as pdf


plt.switch_backend('QT5Agg')


local_repo_path = os.getcwd()

root = Tk()
root.filename = filedialog.askdirectory(title='Select folder containing documents to extract training regions from')
doc_path = root.filename

os.chdir(doc_path)


def func(filename):

    def func2(filename2=filename):
        print(filename)
        doc2 = pdf.convert_from_path("{}.pdf".format(filename2), 500)
        print(doc2[0])
        doc2[0].save("{}.png".format(filename2))
        doc = cv2.imread("{}.png".format(filename2))
        print(doc)
        fig, ax = plt.subplots()
        plt.imshow(doc)

        coords = pd.DataFrame(columns=['blx', 'bly', 'trx', 'try'])

        TrainingRegions = []

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
        plt.tight_layout()
        plt.show()
        for i in range(0, len(coords)):
            TrainingRegions.append(doc[coords.at[i, 'blx']:coords.at[i, 'trx'], coords.at[i, 'bly']:coords.at[i, 'try']])
        # Change from TestImagesTemp to TestImages when function is working
        os.chdir(local_repo_path + '\Data\TestImagesTemp')
        if not coords.empty:
            for j in range(0, len(TrainingRegions)):
                sm.imsave('{}_{}.png'.format(filename2, j), TrainingRegions[j])
    func2()
    return


files = [os.path.splitext(filename)[0] for filename in os.listdir(doc_path)]

for file in files:
    # print(file)
    func(file)
