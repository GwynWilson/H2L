import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.widgets import RectangleSelector, Cursor
import cv2
import os
import scipy.misc as sm
from tkinter import filedialog, Tk
import pdf2image as pdf

positive_response = ['Yes', 'yes', 'y', 'yeah', 'Yeah', 'Y']

plt.switch_backend('QT5Agg')


local_repo_path = os.getcwd()

root = Tk()
root.filename = filedialog.askdirectory(title='Select folder containing documents to extract training regions from')
doc_path = root.filename

os.chdir(doc_path)

ix = -1

initial_check = input("Overwrite any pre-existing training images?")


def func(filename):
    ix = -1

    def func2(filename2=filename):
        print(filename)
        doc2 = pdf.convert_from_path("{}.pdf".format(filename2))
        # print(doc2[0])
        for page in doc2:
            page.save("{}.png".format(filename2))
        doc = cv2.imread("{}.png".format(filename2))
        print(doc)
        fig, ax = plt.subplots()
        plt.imshow(doc, aspect='auto')
        plt.title(filename2)

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
        for i in range(1, len(coords)):
            TrainingRegions.append(doc[coords.at[i, 'bly']:coords.at[i, 'try'], coords.at[i, 'blx']:coords.at[i, 'trx']])
        # Change from TestImagesTemp to TestImages when function is working
        os.chdir(local_repo_path + '\Data\TestImagesTemp')
        if not coords.empty:
            for j in range(0, len(TrainingRegions)):
                sm.imsave('{}__{}.png'.format(filename2, j), TrainingRegions[j])
        os.chdir(doc_path)
        os.remove("{}.png".format(filename2))
    func2()
    return


files = [os.path.splitext(filename)[0] for filename in os.listdir(doc_path)]
preexisting_files = os.listdir(local_repo_path + "\Data\TestImagesTemp")
print(preexisting_files)
preexisting_files_split = [preexisting_files[i].split('__')[0] for i in range(0, len(preexisting_files))]
print(preexisting_files_split)

for file in files:
    print(file)
    if file.split('__')[0] in preexisting_files_split:
        if initial_check not in positive_response:
            continue
    func(file)
