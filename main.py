import os
import numpy as np
import cv2
from tqdm import tqdm
from utils.util import histogram_data, tonemapping, get_dynamic_range


dataset = 'HDR-Eye'
input_path = f'./dataset/{dataset}/'
save_path = f'./result/{dataset}/'
draw_histogram_image = True
GT_to_bit = '8bit'
tonemapping_method = 'Reinhard'
os.mkdir


def get_histogram_data(image_path, result_save_path, bit, draw_histogram_image):

    dynamic_range, _ = get_dynamic_range(bit)
    histogram_data(image_path, result_save_path,
                   dynamic_range, draw_histogram_image)


if __name__ == '__main__':

    # Input data
    print('Input')

    data_path = input_path+dataset+'-input/'
    result_save_path = save_path+dataset+'-input/'

    if not os.path.isdir(result_save_path):
        os.makedirs(result_save_path)

    for filename in tqdm(os.listdir(data_path)):

        # get histogram data
        image_path = data_path + filename
        result_save_name = result_save_path+filename[:-4]
        get_histogram_data(image_path, result_save_name,
                           '8bit', draw_histogram_image)

    print('GT')

    # path
    data_path = input_path+dataset+'-gt/'

    result_save_path = save_path+dataset+'-gt/histogram/'
    tonemapping_save_path = save_path+dataset+'-gt/tonemapping/'

    # check dir
    if not os.path.isdir(result_save_path):
        os.makedirs(result_save_path)
    if not os.path.isdir(tonemapping_save_path):
        os.makedirs(tonemapping_save_path)

    for filename in tqdm(os.listdir(data_path)):
        image_path = data_path + filename
        tonemapping_save_name = tonemapping_save_path+filename[:-4]

        # tonemapping
        imBatch = tonemapping(image_path, GT_to_bit, tonemapping_method)
        tonemapping_save_name += ".png"
        cv2.imwrite(tonemapping_save_name, imBatch)

        # get histogram data
        result_save_name = result_save_path+filename[:-4]
        get_histogram_data(tonemapping_save_name, result_save_name,
                           GT_to_bit, draw_histogram_image)
