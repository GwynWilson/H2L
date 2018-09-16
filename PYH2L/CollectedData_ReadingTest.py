import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
print(os.getcwd())
# If your project is contained in the same type of IdeaProjects form as mine, this file path should work.
test = pd.read_csv("Data\MergeDat\Ting.csv", engine='c')
test = test.drop(['Unnamed: 2'], axis=1).set_index(['Unnamed: 0', 'Unnamed: 1']).rename_axis(('Source',
                                                                                              'Type'))
plt.switch_backend('QT5Agg')

image_debugging = input("\n Debug by plotting? \n\n \t0 - No debugging \n \t1 - Plot file outputs \n \t2 - Plot function outputs \n \t3 - Plot everything \n\n "
                        "Input:")


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
        if int(image_debugging) >= 3:
            plt.imshow(Oriented_Img_Data)
            plt.title('Output of extract_data')
            plt.show()
        return Oriented_Img_Data, Shaped_Coords
    elif not keepna:
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
        if int(image_debugging) >= 3:
            plt.imshow(Oriented_Img_Data)
            plt.title('Output of extract_data')
            plt.show()
        return Oriented_Img_Data, Shaped_Coords


"""
Creating empty lists to use in prepare_data. As they are resized to have uniform dimensions the list can be converted to a multidimension numpy array as 
supposed to a numpy array containing array objects.
"""
train_imgs = []
train_coords = []
test_imgs = []
test_coords = []
train_boolmap = []
test_boolmap = []

# TODO: Resize images to reduce resolution
# TODO: Fix splitting of training and testing data


def prepare_data():
    """
    Resizes all img arrays to have the same dimensions, similarly for all coord arrays. Appends the returned data of extract_data for each layer contained in
    the dataset.
    :return: Separate training and test arrays for both imgs and coords.
    :rtype: np.array
    """
    global train_imgs, test_imgs, train_boolmap, test_boolmap, train_coords, test_coords
    max_width = 0
    max_length = 0
    max_coord_length = 0
    levels_index = list(range(0, len(test.index.levels[0])))
    train_index = levels_index[:(len(levels_index)//2)+1]
    test_index = levels_index[(len(levels_index)//2):]
    for n in train_index:
        img, coord = extract_data(n, True)
        if int(image_debugging) >= 2:
            plt.imshow(img)
            plt.title('Input into prepare_data from extract_data')
            plt.show()
        img[img == 0] = 255.
        """
        Checks if the current img and coord array of the loop is the largest array so far, if so notes the length (for img data) and width (for coord data). 
        The fact that length is used for img data and width is used for coord data is fairly arbitrary and very much a result of this being a work in 
        progress. I'm currently thinking that shaping each array to be a square with the lengths equal to the largest dimension (either length or width) of 
        any of the arrays will be best for ensuring that keras can read the data easily. However, this will increase the size of the tensors and there are 
        probably more efficient ways to shape the arrays.
        """
        if img.shape[1] >= max_width:
            max_width = img.shape[1]
        if img.shape[0] >= max_length:
            max_length = img.shape[0]
        if coord.shape[0] >= max_coord_length:
            max_coord_length = coord.shape[0]
        if int(image_debugging) >= 2:
            plt.imshow(img)
            plt.show()
        train_imgs.append(np.array(img))
        train_coords.append(coord)
    for m in test_index:
        img, coord_2 = extract_data(m, True)
        if int(image_debugging) >= 2:
            plt.imshow(img)
            plt.title('Input into prepare_data from extract_data')
            plt.show()
        img[img == 0] = 255.
        if img.shape[1] >= max_width:
            max_width = img.shape[1]
        if img.shape[0] >= max_length:
            max_length = img.shape[0]
        if coord_2.shape[0] >= max_coord_length:
            max_coord_length = coord_2.shape[0]
        test_imgs.append(np.array(img))
        test_coords.append(coord_2)
    if max_length % 2 != 0:
        max_length += 1
    if max_width % 2 != 0:
        max_width += 1
    dimensions = (max_length, max_width)

    width_dif = [abs(dimensions[0] - train_imgs[i].shape[0]) for i in range(0, len(train_imgs))]
    height_dif = [abs(dimensions[1] - train_imgs[i].shape[1]) for i in range(0, len(train_imgs))]
    for i in range(0, len(train_imgs)):
        train_imgs[i] = np.pad(train_imgs[i], ((1, width_dif[i]), (1, height_dif[i])), mode='constant', constant_values=255)

    width_dif2 = [abs(dimensions[0] - test_imgs[i].shape[0]) for i in range(0, len(test_imgs))]
    height_dif2 = [abs(dimensions[1] - test_imgs[i].shape[1]) for i in range(0, len(test_imgs))]
    for i in range(0, len(test_imgs)):
        test_imgs[i] = np.pad(test_imgs[i], ((1, width_dif2[i]), (1, height_dif2[i])), mode='constant', constant_values=255)

    for i in range(0, len(train_coords)):
        train_coords[i] = np.pad(train_coords[i], ((0, abs(len(train_coords[i])-max_coord_length)), (0, 0)), mode='constant', constant_values=0)
    for i in range(0, len(test_coords)):
        test_coords[i] = np.pad(test_coords[i], ((0, abs(len(test_coords[i])-max_coord_length)), (0, 0)), mode='constant', constant_values=0)

    if int(image_debugging) >= 1:
        for i in range(0, len(train_imgs)):
            plt.imshow(train_imgs[i])
            for f in range(0, len(train_coords[i])):
                bot_left_x = min(train_coords[i][f][0], train_coords[i][f][2])
                bot_left_y = min(train_coords[i][f][1], train_coords[i][f][3])
                width = abs(train_coords[i][f][0]-train_coords[i][f][2])
                height = abs(train_coords[i][f][1]-train_coords[i][f][3])
                plt.gca().add_patch(matplotlib.patches.Rectangle((bot_left_x, bot_left_y), width,
                                                                 height, ec='r', fc='none', lw=3))
            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.showMaximized()
            plt.show()
    """
    I am rounding to reduce space. However, it is odd that at this point the arrays contain floating point values as all of the data should be integers.
    """
    train_imgs3 = np.round(np.array(train_imgs), 3)
    train_coords3 = np.array(train_coords)
    test_imgs3 = np.round(np.array(test_imgs), 3)
    test_coords3 = np.array(test_coords)
    test_boolmap2 = np.array(test_boolmap)
    train_boolmap2 = np.array(train_boolmap)
    print('Data prepared')
    return train_imgs3, test_imgs3, test_boolmap2, train_boolmap2, dimensions, train_coords3, test_coords3
