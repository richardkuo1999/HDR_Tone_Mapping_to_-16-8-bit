# import os
# import numpy as np
# import cv2
# from utils.util import load_data, loadHdr, tonemapping


dataset = 'HDR-Eye'
input_path = f'./dataset/{dataset}/'
save_path = './result/'

print(input_path)
# if __name__ == '__main__':

#     for i in os.listdir(input_path):
#         print(i)

#         imBatch = loadHdr(input_path+i)
#         cv2.imwrite(save_path+"image/"+i[:-4] + ".png", imBatch)

#         load_data(save_path+"image/"+i[:-4] +
#                   ".png", save_path+"result/"+i[:-4])
