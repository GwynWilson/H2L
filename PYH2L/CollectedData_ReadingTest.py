import numpy as np
import pandas as pd
from skimage.transform import resize, rescale
import os

import matplotlib.pyplot as plt
print(os.getcwd())
# If your project is contained in the same type of IdeaProjects form as mine, this file path should work.
test = pd.read_csv("Data\MergeDat\Ting.csv", engine='c')
test = test.drop(['Unnamed: 2'], axis=1).set_index(['Unnamed: 0', 'Unnamed: 1']).rename_axis(('Source',
                                                                                              'Type'))


def rebin(a, shape):
    """

    :param a: Array to reshape
    :type a: np.array or other array object
    :param shape: Dimensions of new array
    :type shape: tuple
    :return: Array with the dimensions: shape
    :rtype:
    """
    sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def extract_data(layer, keepna):
    """

    :param layer: The level (image/coordinate set) of the MultiIndex 'test' to process.
    :type layer: int
    :param keepna: Whether to drop the empty cells to the right of a given layer or to keep them for the purpose of having all levels have the same dimensions.
    :type keepna: bool
    :return: Array of img data, array of coordinate data.
    :rtype: np.array
    """
    layers = test.index.levels[0]
    layer_name = layers[layer]
    Img = test.loc[(slice(None), 'Img'), :]
    Coords = test.loc[(slice(None), 'Coords'), :]
    """
    There are lots of steps here, none of them are particularly exciting. I don't think that I can give any more clarity in what each step does than you 
    would get from printing each step.
    """
    if not keepna:
        Img_Array_Shape_Mask = Img.xs(layer_name).isna()
        Img_Shape = Img_Array_Shape_Mask[~Img_Array_Shape_Mask].dropna(axis=1).shape
        Coords_Array_Shape_Mask = Coords.xs(layer_name).isna()
        Coords_Shape = Coords_Array_Shape_Mask[~Coords_Array_Shape_Mask].dropna(axis=1).shape
        Imgs_Array = Img.groupby(level=0).apply(lambda x: x.values.tolist()).values
        Coords_Array = Coords.groupby(level=0).apply(lambda x: x.values.tolist()).values
        Imgs_Array = [np.extract(np.logical_not(np.isnan(Imgs_Array[i])), Imgs_Array[i]) for i in range(0, len(Imgs_Array))]
        Coords_Array = [np.extract(np.logical_not(np.isnan(Coords_Array[i])), Coords_Array[i]) for i in range(0, len(Coords_Array))]
        Oriented_Img_Data = np.array(np.fliplr(np.rot90(np.array(Imgs_Array[layer]).reshape(Img_Shape).astype("uint8"), 3)))
        Shaped_Coords = np.array(np.array(Coords_Array[layer]).reshape(Coords_Shape))
        return Oriented_Img_Data, Shaped_Coords
    if keepna:
        Img_Shape = Img.xs(layer_name).shape
        Coords_Array_Shape_Mask = Coords.xs(layer_name).isna()
        Coords_Shape = Coords_Array_Shape_Mask[~Coords_Array_Shape_Mask].dropna(axis=1).shape
        Imgs_Array = Img.groupby(level=0).apply(lambda x: x.values.tolist()).values
        Coords_Array = Coords.groupby(level=0).apply(lambda x: x.values.tolist()).values
        Coords_Array = [np.extract(np.logical_not(np.isnan(Coords_Array[i])), Coords_Array[i]) for i in range(0, len(Coords_Array))]
        Oriented_Img_Data = np.array(np.fliplr(np.rot90(np.array(Imgs_Array[layer]).reshape(Img_Shape).astype(
                "uint8"), 3)))
        Shaped_Coords = np.array(np.array(Coords_Array[layer]).reshape(Coords_Shape))
        # print('SHAPED COORDS', Shaped_Coords)
        return Oriented_Img_Data, Shaped_Coords


"""
Creating empty lists to use in prepare_data. As they are resized to have uniform dimensions the list can be converted to a multidimension numpy array as 
supposed to a numpy array containing array objects.
"""
train_imgs = []
train_coords = []
test_imgs = []
test_coords = []
train_coords2 = []
test_coords2 = []


def prepare_data():
    """
    Resizes all img arrays to have the same dimensions, similarly for all coord arrays. Appends the returned data of extract_data for each layer contained in
    the dataset.
    :return: Separate training and test arrays for both imgs and coords.
    :rtype: np.array
    """
    global train_imgs, test_imgs, train_coords, test_coords, train_coords2, test_coords2
    max_length = 0
    max_coord_length = 0
    levels_index = list(range(0, len(test.index.levels[0])))
    train_index = levels_index[:len(levels_index)//2]
    test_index = levels_index[len(levels_index)//2:]
    for n in train_index:
        img, coord = extract_data(n, True)
        coord2 = np.zeros(img.shape, dtype=int)
        for x in range(0, len(coord)):
            coord2[int(coord[x][0]):int(coord[x][2]), int(coord[x][1]):int(coord[x][3])] = 1
        coord[np.isnan(coord)] = 0
        img[np.isnan(img)] = 0
        # print('IMAGE SHAPE', img.shape)
        # print('COORD SHAPE', coord.shape)
        """
        Checks if the current img and coord array of the loop is the largest array so far, if so notes the length (for img data) and width (for coord data). 
        The fact that length is used for img data and width is used for coord data is fairly arbitrary and very much a result of this being a work in 
        progress. I'm currently thinking that shaping each array to be a square with the lengths equal to the largest dimension (either length or width) of 
        any of the arrays will be best for ensuring that keras can read the data easily. However, this will increase the size of the tensors and there are 
        probably more efficient ways to shape the arrays.
        """
        if img.shape[1] >= max_length:
            max_length = img.shape[1]
        if coord.shape[0] >= max_coord_length:
            max_coord_length = coord.shape[0]
        coord2 = np.fliplr(np.rot90(coord2, 3))
        # plt.imshow(img)
        # plt.imshow(coord2, alpha=0.1)
        # plt.show()
        train_coords2.append(coord2)
        train_imgs.append(np.array(img))
        train_coords.append(coord)
    for m in test_index:
        img, coord_2 = extract_data(m, True)
        coord2 = np.zeros(img.shape, dtype=int)
        for z in range(0, len(coord_2)):
            coord2[int(coord_2[z][0]):int(coord_2[z][2]), int(coord_2[z][1]):int(coord_2[z][3])] = 1
        coord_2[np.isnan(coord_2)] = 0
        img[np.isnan(img)] = 0
        # print('IMAGE SHAPE', img.shape)
        # print('COORD SHAPE', coord_2.shape)
        if img.shape[1] >= max_length:
            max_length = img.shape[1]
        if coord_2.shape[0] >= max_coord_length:
            max_coord_length = coord_2.shape[0]
        coord2 = np.fliplr(np.rot90(coord2, 3))
        # plt.imshow(img)
        # plt.imshow(coord2, alpha=0.1)
        # plt.show()
        test_coords2.append(coord2)
        test_imgs.append(np.array(img))
        test_coords.append(coord_2)
    # print('TRAIN COORDS, PRE-RE-SIZING', train_coords)
    # print('MAXIMUM LENGTHS', max_length, max_coord_length)
    """
    Here skimage.transform.resize is used to carry out the transformation. The documentation for this function is online and should provide all necessary 
    insight into the function arguments used. List comprehension is used to iterate over each layer within the new array of arrays.
    """
    train_imgs = [resize(train_imgs[n], (313, max_length), mode='constant', cval=0) for n in range(0, len(train_imgs))]
    test_imgs = [resize(test_imgs[n], (313, max_length), mode='constant', cval=0) for n in range(0, len(test_imgs))]
    train_coords = [resize(train_coords[n], (max_coord_length, 4), mode='constant', cval=0) for n in range(0,
                                                                                                        len(train_coords))]
    test_coords = [resize(test_coords[n], (max_coord_length, 4), mode='constant', cval=0) for n in range(0,
                                                                                                      len(test_coords))]
    train_coords2 = [resize(train_coords2[n], (313, max_length), mode='constant', cval=0) for n in range(0,
                                                                                                           len(train_coords2))]

    test_coords2 = [resize(test_coords2[n], (313, max_length), mode='constant', cval=0) for n in range(0,
                                                                                                         len(test_coords2))]
    # train_coords2 = [rescale(train_coords2[n], 0.5) for n in range(0, len(train_coords2))]
    # test_coords2 = [rescale(test_coords2[n], 0.5) for n in range(0, len(test_coords2))]
    # train_imgs = [rescale(train_imgs[n], 0.5) for n in range(0, len(train_imgs))]
    # test_imgs = [rescale(test_imgs[n], 0.5) for n in range(0, len(test_imgs))]
    for h in range(0, len(test_coords2)):
        plt.imshow(test_imgs[h])
        plt.imshow(test_coords2[h], alpha=0.1)
        plt.show()
    for h in range(0, len(train_coords2)):
        plt.imshow(train_imgs[h])
        plt.imshow(train_coords2[h], alpha=0.1)
        plt.show()
    # train_coords = [resize(train_coords[n], (313, max_length), mode='constant', cval=0) for n in range(0, len(train_coords))]
    # test_coords = [resize(test_coords[n], (313, max_length), mode='constant', cval=0) for n in range(0, len(test_coords))]
    """
    I am rounding to reduce space. However, it is odd that at this point the arrays contain floating point values as all of the data should be integers.
    """
    train_imgs3 = np.round(np.array(train_imgs), 3)
    train_coords3 = np.round(np.array(train_coords), 0)
    test_imgs3 = np.round(np.array(test_imgs), 3)
    test_coords3 = np.round(np.array(test_coords), 0)
    test_coords4 = np.array(test_coords2)
    train_coords4 = np.array(train_coords2)
    # print('train_imgs', train_imgs3, '\n')
    # print('train_coords', train_coords3, '\n')
    # print('test_imgs', test_imgs3, '\n')
    # print('test_coords', test_coords3, '\n')
    # print('train_coords4', train_coords4, '\n')
    # print('test_coords4', test_coords4, '\n')
    return train_imgs3, train_coords3, test_imgs3, test_coords3, test_coords4, train_coords4


prepare_data()