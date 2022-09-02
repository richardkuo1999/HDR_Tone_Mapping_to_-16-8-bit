import os
import numpy as np
import cv2
# from utils.util import extend_upsample_downsample, draw_hist_cdf
from matplotlib import pyplot as plt

input_path = './data/hdr/'
save_path = './result/'


def load_data(input_path, save_pat):

    # Input color image
    # print("[INFO] Reading and converting image......")
    IN_img = cv2.imread(input_path, cv2.IMREAD_ANYDEPTH)  # uint8
    # GT_img = cv2.imread(
    #     grounp_true_path, cv2.IMREAD_UNCHANGED)  # uint16
    print(IN_img.max())
    # img = cv2.imread("gt.hdr", cv2.IMREAD_ANYDEPTH)
    dynamic_range = 70000
    hist, bins = np.histogram(
        IN_img.flatten(), dynamic_range, [0, dynamic_range])
    #hist = scipy.signal.savgol_filter(hist,33, 3) #平滑化 ###
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    fig = plt.figure()
    plt.plot(cdf_normalized, color='b')
    # plt.bar(range(0, dynamic_range), hist, color='r', width=1)
    plt.plot(range(0, dynamic_range), hist, marker='o', markersize=0.00001)
    plt.xlim([0, dynamic_range])
    plt.ylim([0, hist.max()+hist.max()*0.1])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    fig.savefig((save_pat+".png"), dpi=fig.dpi)
    plt.close()
    np.savetxt(save_pat+".txt", hist, fmt='%d', newline=' ')


def loadHdr(imName):
    img = cv2.imread(imName, flags=cv2.IMREAD_ANYDEPTH)

    tonemapDurand = cv2.createTonemapReinhard(2.2, 0, 0, 0)

    ldrDurand = tonemapDurand.process(img)

    im2_16bit = np.clip(ldrDurand * 65536, 0, 65536).astype('uint16')
    # 255 65536 4294967296
    return im2_16bit


if __name__ == '__main__':
    # 要改rgb_mode,txt的路徑(2+1)
    # rgb_mode = 2 #改

    for i in os.listdir(input_path):
        print(i)

        imBatch = loadHdr(input_path+i)
        cv2.imwrite(save_path+"image/"+i[:-4] + ".png", imBatch)

        load_data(save_path+"image/"+i[:-4] +
                  ".png", save_path+"result/"+i[:-4])
