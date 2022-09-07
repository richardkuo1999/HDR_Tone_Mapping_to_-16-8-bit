import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse
from utils.util import tonemapping, get_histogram_data


parser = argparse.ArgumentParser()
train_set = parser.add_mutually_exclusive_group()

# _/_/_/ what to do _/_/_/
parser.add_argument('--dataset', default='HDR-Eye',
                    help='dataset file name',
                    type=str)
parser.add_argument('--input_path', default=f'./dataset/',
                    help='input path',
                    type=str)
parser.add_argument('--save_path', default=f'./result/',
                    help='save path',
                    type=str)
parser.add_argument('--draw_histogram_image', default=True,
                    help='whether draw histogram image',
                    type=bool)
parser.add_argument('--GT_to_bit', default='8bit',
                    choices=['8bit,16bit'],
                    help='GT to what bit 8bit or 16bit',
                    type=str)
parser.add_argument('--tonemapping_method', default='Reinhard',
                    choices=['Reinhard', 'Drago', 'Durand', 'Mantiuk'],
                    help='choice tonemapping method ',
                    type=str)

args = parser.parse_args()


if __name__ == '__main__':

    args.input_path += f'{args.dataset}/'
    args.save_path += f'{args.dataset}/'

    # Input data
    print('Input')

    data_path = args.input_path+args.dataset+'-input/'
    result_save_path = args.save_path+args.dataset+'-input/'

    if not os.path.isdir(result_save_path):
        os.makedirs(result_save_path)
    # data by data
    for filename in tqdm(os.listdir(data_path)):

        # get histogram data
        image_path = data_path + filename
        result_save_name = result_save_path+filename[:-4]
        get_histogram_data(image_path, result_save_name,
                           '8bit', args.draw_histogram_image)

    print('GT')

    # path
    data_path = args.input_path+args.dataset+'-gt/'

    result_save_path = args.save_path+args.dataset+'-gt/histogram/'
    tonemapping_save_path = args.save_path+args.dataset+'-gt/tonemapping/'

    # check dir
    if not os.path.isdir(result_save_path):
        os.makedirs(result_save_path)
    if not os.path.isdir(tonemapping_save_path):
        os.makedirs(tonemapping_save_path)
    # data by data
    for filename in tqdm(os.listdir(data_path)):
        image_path = data_path + filename
        tonemapping_save_name = tonemapping_save_path+filename[:-4]

        # tonemapping
        imBatch = tonemapping(image_path, args.GT_to_bit,
                              args.tonemapping_method)
        tonemapping_save_name += ".png"
        cv2.imwrite(tonemapping_save_name, imBatch)

        # get histogram data
        result_save_name = result_save_path+filename[:-4]
        get_histogram_data(tonemapping_save_name, result_save_name,
                           args.GT_to_bit, args.draw_histogram_image)
