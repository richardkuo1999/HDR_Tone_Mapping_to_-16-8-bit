import os
import numpy as np
import cv2
from tqdm import tqdm
from utils.util import histogram_data, loadHdr, tonemapping


dataset = 'HDR-Eye'
input_path = f'./dataset/{dataset}/'
save_path = f'./result/{dataset}/'
draw_histogram_image = True
GT_to_bit = '16bit'

os.mkdir


def get_histogram_data(image_path, result_save_path, bit, draw_histogram_image):
    if bit == '8bit':
        dynamic_range = 255
    elif bit == '16bit':
        dynamic_range = 65536

    histogram_data(image_path, result_save_path,
                   dynamic_range, draw_histogram_image)


if __name__ == '__main__':

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Input data
    print('Input')

    data_path = input_path+dataset+'-input/'
    result_save_path = save_path+dataset+'-input/'

    if not os.path.isdir(result_save_path):
        os.makedirs(result_save_path)

    for filename in tqdm(os.listdir(data_path)):

        image_path = data_path + filename
        result_save_name = result_save_path+filename[:-4]
        get_histogram_data(image_path, result_save_name,
                           '8bit', draw_histogram_image)

    print('GT')

    data_path = input_path+dataset+'-gt/'
    result_save_path = save_path+dataset+'-gt/'

    if not os.path.isdir(result_save_path):
        os.makedirs(result_save_path)

    for filename in tqdm(os.listdir(data_path)):
        image_path = data_path + filename
        result_save_name = result_save_path+filename[:-4]

        get_histogram_data(image_path, result_save_name,
                           '8bit', draw_histogram_image)
        #         imBatch = loadHdr(input_path+i)
        #         cv2.imwrite(save_path+"image/"+i[:-4] + ".png", imBatch)

        #         load_data(save_path+"image/"+i[:-4] +
        #                   ".png", save_path+"result/"+i[:-4])
