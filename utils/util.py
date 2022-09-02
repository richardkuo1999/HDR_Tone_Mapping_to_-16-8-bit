import numpy as np
import cv2
from matplotlib import pyplot as plt


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


def loadHdr(imName):
    img = cv2.imread(imName, flags=cv2.IMREAD_ANYDEPTH)

    tonemapDurand = cv2.createTonemapReinhard(2.2, 0, 0, 0)

    ldrDurand = tonemapDurand.process(img)

    im2_16bit = np.clip(ldrDurand * 65536, 0, 65536).astype('uint16')
    # 255 65536 4294967296
    return im2_16bit


def tonemapping(img, dynamic_range, save_path, color_space, tag=None, method='Reinhard'):
    # float 32 bits 3 channels
    img = (img/(dynamic_range-1)).astype('float32')  # 0~1

    if img.ndim != 3:
        img = cv2.merge([img, img, img])
    if method == 'Reinhard':
        tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
        ldrReinhard = tonemapReinhard.process(img)
        cv2.imwrite(save_path + tag + color_space +
                    "_TMresult.png", ldrReinhard * 255)
    elif method == 'Drago':
        tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
        ldrDrago = tonemapDrago.process(img)
        ldrDrago = 3 * ldrDrago
        cv2.imwrite(save_path + tag + color_space +
                    "_TMresult.png", ldrDrago * 255)
    elif method == 'Durand':
        tonemapDurand = cv2.createTonemapDurand(1.5, 4, 1.0, 1, 1)
        ldrDurand = tonemapDurand.process(img)
        ldrDurand = 3 * ldrDurand
        cv2.imwrite(save_path + tag + color_space +
                    "_TMresult.png", ldrDurand * 255)
    elif method == 'Mantiuk':
        tonemapMantiuk = cv2.createTonemapMantiuk(2.2, 0.85, 1.2)
        ldrMantiuk = tonemapMantiuk.process(img)
        ldrMantiuk = 3 * ldrMantiuk
        cv2.imwrite(save_path + tag + color_space +
                    "_TMresult.png", ldrMantiuk * 255)
    else:
        print("Please enter the tonemapping method!")
