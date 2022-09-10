import numpy as np
import cv2
from matplotlib import pyplot as plt


def get_dynamic_range(bit):
    if bit == '8bit':
        dynamic_range = 256
        astype = 'uint8'
    elif bit == '16bit':
        dynamic_range = 65536
        astype = 'uint16'
    return dynamic_range, astype


def histogram_data(image_path, save_path, dynamic_range, draw_histogram_image):

    # Input color image
    IN_img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)

    hist, _ = np.histogram(
        IN_img.flatten(), dynamic_range, [0, dynamic_range])

    # draw_histogram_image
    if (draw_histogram_image):
        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / cdf.max()

        fig = plt.figure()
        plt.plot(cdf_normalized, color='b')
        plt.plot(range(0, dynamic_range), hist, marker='o', markersize=0.00001)
        plt.xlim([0, dynamic_range])
        if (hist.min() >= 0):
            plt.ylim([0, hist.max()+hist.max()*0.1])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        fig.savefig((save_path+".png"), dpi=fig.dpi)
        plt.close()

    np.savetxt(save_path+".txt", hist, fmt='%d', newline=' ')


def tonemapping(imName, GT_to_bit, method='Reinhard'):

    dynamic_range, astype = get_dynamic_range(GT_to_bit)
    img = cv2.imread(imName, flags=cv2.IMREAD_ANYDEPTH)

    if method == 'Reinhard':
        tonemap = cv2.createTonemapReinhard(2.2, 0, 0, 0)
    elif method == 'Drago':
        tonemap = cv2.createTonemapDrago(1.0, 0.7)
    elif method == 'Durand':
        tonemap = cv2.createTonemapDurand(1.5, 4, 1.0, 1, 1)
    elif method == 'Mantiuk':
        tonemap = cv2.createTonemapMantiuk(2.2, 0.85, 1.2)
    else:
        print("Please enter the tonemapping method!")
    ldr = tonemap.process(img)
    im2 = np.clip(ldr * dynamic_range, 0, dynamic_range).astype(astype)
    # 256 65536 4294967296
    return im2


def get_histogram_data(image_path, result_save_path, bit, draw_histogram_image):

    dynamic_range, _ = get_dynamic_range(bit)
    histogram_data(image_path, result_save_path,
                   dynamic_range, draw_histogram_image)
