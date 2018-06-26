import numpy as np
import pandas as pd
from skimage.transform import resize
test = pd.read_csv("C:\\Users\Josh\IdeaProjects\H2L\PYH2L\Data\MergeDat\Ting.csv", engine='c')
test = test.drop(['Unnamed: 2'], axis=1).set_index(['Unnamed: 0', 'Unnamed: 1']).rename_axis(('Source',
                                                                                              'Type'))


def extract_data(layer, keepna):
    layers = test.index.levels[0]
    layer_name = layers[layer]
    Img = test.loc[(slice(None), 'Img'), :]
    Coords = test.loc[(slice(None), 'Coords'), :]
    if not keepna:
        Img_Array_Shape_Mask = Img.xs(layer_name).isna()
        Img_Shape = Img_Array_Shape_Mask[~Img_Array_Shape_Mask].dropna(axis=1).shape
        Coords_Array_Shape_Mask = Coords.xs(layer_name).isna()
        Coords_Shape = Coords_Array_Shape_Mask[~Coords_Array_Shape_Mask].dropna(axis=1).shape
        Imgs_Array = Img.groupby(level=0).apply(lambda x: x.values.tolist()).values
        Coords_Array = Coords.groupby(level=0).apply(lambda x: x.values.tolist()).values
        Imgs_Array = [np.extract(np.logical_not(np.isnan(Imgs_Array[i])), Imgs_Array[i]) for i in range(0, len(Imgs_Array))]
        Coords_Array = [np.extract(np.logical_not(np.isnan(Coords_Array[i])), Coords_Array[i]) for i in range(0, len(Coords_Array))]
        # print(np.array(Imgs_Array[layer]), '\n\n')
        Oriented_Img_Data = np.array(np.fliplr(np.rot90(np.array(Imgs_Array[layer]).reshape(Img_Shape).astype("uint8"), 3)))
        Shaped_Coords = np.array(np.array(Coords_Array[layer]).reshape(Coords_Shape))
        return Oriented_Img_Data, Shaped_Coords
    if keepna:
        Img_Shape = Img.xs(layer_name).shape
        Coords_Shape = Coords.xs(layer_name).shape
        Imgs_Array = Img.groupby(level=0).apply(lambda x: x.values.tolist()).values
        Coords_Array = Coords.groupby(level=0).apply(lambda x: x.values.tolist()).values
        # print(np.array(Imgs_Array[layer]), '\n\n')
        Oriented_Img_Data = np.array(np.fliplr(np.rot90(np.array(Imgs_Array[layer]).reshape(Img_Shape).astype("uint8"), 3)))
        Shaped_Coords = np.array(np.array(Coords_Array[layer]).reshape(Coords_Shape))
        return Oriented_Img_Data, Shaped_Coords


train_imgs = []
train_coords = []
test_imgs = []
test_coords = []


def prepare_data():
    global train_imgs, test_imgs, train_coords, test_coords
    max_length = 0
    levels_index = list(range(0, len(test.index.levels[0])))
    train_index = levels_index[:len(levels_index)//2]
    test_index = levels_index[len(levels_index)//2:]
    for n in train_index:
        img, coord = extract_data(n, True)
        coord[np.isnan(coord)] = 0
        img[np.isnan(img)] = 0
        print(img.shape)
        if img.shape[1] >= max_length:
            max_length = img.shape[1]
        train_imgs.append(np.array(img))
        train_coords.append(np.array(coord))
    for m in test_index:
        img, coord = extract_data(m, True)
        coord[np.isnan(coord)] = 0
        img[np.isnan(img)] = 0
        print(img.shape)
        if img.shape[1] >= max_length:
            max_length = img.shape[1]
        test_imgs.append(np.array(img))
        test_coords.append(np.array(coord))
    print(max_length)
    train_imgs = [resize(train_imgs[n], (313, max_length), mode='constant', cval=0) for n in range(0, len(train_imgs))]
    test_imgs = [resize(test_imgs[n], (313, max_length), mode='constant', cval=0) for n in range(0, len(test_imgs))]
    train_coords = [resize(train_coords[n], (313, max_length), mode='constant', cval=0) for n in range(0, len(train_coords))]
    test_coords = [resize(test_coords[n], (313, max_length), mode='constant', cval=0) for n in range(0, len(test_coords))]
    print('train_imgs', np.array(train_imgs), '\n')
    print('train_coords', np.array(train_coords), '\n')
    print('test_imgs', np.array(test_imgs), '\n')
    print('test_coords', np.array(test_coords), '\n')
    return np.array(train_imgs), np.array(train_coords), np.array(test_imgs), np.array(test_coords)


prepare_data()
