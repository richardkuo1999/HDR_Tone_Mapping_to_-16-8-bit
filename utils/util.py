import numpy as np
import cv2
from matplotlib import pyplot as plt
# import tensorflow as tf #ssim
import argparse
import scipy.signal
import math


def extend_upsample_downsample(IN_img, IN_range, TA_range, IN_dtype, TA_dtype, save_path, colors_name, draw_hist=True, save_img=True):
    height, width = IN_img.shape
    upH = height * int(math.sqrt(TA_range/IN_range))
    upW = width * int(math.sqrt(TA_range/IN_range))
    # Extend
    #print("[INFO] Extending......")
    # IN_extend = IN_img.astype(TA_dtype) #add
    # IN_extend = (IN_extend * (TA_range/IN_range)).astype(TA_dtype) #'uint16'
    IN_extend = (IN_img * (TA_range/IN_range)).astype(TA_dtype)  # 'uint16'
    #_, _ = draw_hist_cdf(IN_extend, TA_range, save_path, colors_name, tag = "extend_")

    # Upsampling
    #print("[INFO] Upsampling......")
    IN_extend = cv2.resize(IN_extend, (upH, upW),
                           interpolation=cv2.INTER_CUBIC)
    #_, _ = draw_hist_cdf(IN_extend, TA_range, save_path, colors_name, tag = "upsample_")

    # Downsampling
    #print("[INFO] Downsampling......")
    IN_extend = cv2.resize(IN_extend, (height, width),
                           interpolation=cv2.INTER_AREA)
    EXT_hist, EXT_cdf = draw_hist_cdf(
        IN_extend, TA_range, save_path, colors_name, tag="downsample_")  # False #True

    return IN_extend, EXT_hist, EXT_cdf


def float2uint(img, dynamic_range, target_dtype):
    img = np.maximum(0, img)
    img = np.minimum(1, img)
    img = (img * (dynamic_range - 1) + 0.5).astype(target_dtype)
    return img


'''
def cvtcolor_12bit_YCrCb2BGR(img):
    # 12bit to 16bit
    img = (np.round(img/4095*65535)).astype('uint16')
    # YCrCb2BGR
    BGR_img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    # 16bit to 12bit
    BGR_img = (np.round(BGR_img/65535*4095)).astype('uint16')
    return BGR_img
'''


def min_max(img):
    if img.ndim == 3:
        return (np.min(img, axis=(0, 1, 2))), (np.max(img, axis=(0, 1, 2)))
    elif img.ndim == 2:
        return (np.min(img, axis=(0, 1))), (np.max(img, axis=(0, 1)))
    elif img.ndim == 1:
        return (np.min(img, axis=0)), (np.max(img, axis=0))
    else:
        print("min_max function input error!")

# input image(1 channel)


def draw_hist_cdf(img, dynamic_range, save_path, colors_name=None, tag=None, draw_hist=True, draw_cdf=True, save_img=True):
    hist, bins = np.histogram(img.flatten(), dynamic_range, [0, dynamic_range])
    #hist = scipy.signal.savgol_filter(hist,33, 3) #平滑化 ###
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    fig = plt.figure()
    # plt.plot(cdf_normalized, color='b')
    # plt.bar(range(0, dynamic_range), hist, color='r', width=1)
    plt.plot(range(0, dynamic_range), hist, marker='o', markersize=0.00001)
    plt.xlim([0, dynamic_range])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    if save_img:
        fig.savefig((save_path + tag + colors_name + "_hist.png"), dpi=fig.dpi)
    plt.close()
    return hist, cdf


def draw_hist(img, dynamic_range, save_path, colors_name=None, tag=None, draw_hist=True, save_img=True):
    hist, bins = np.histogram(img.flatten(), dynamic_range, [0, dynamic_range])
    fig = plt.figure()
    plt.bar(range(0, dynamic_range), hist, color='r', width=1)
    plt.xlim([0, dynamic_range])
    plt.legend(('histogram'), loc='upper left')
    if save_img:
        fig.savefig((save_path + tag + colors_name +
                    "_onlyhist.png"), dpi=fig.dpi)
    plt.close()
    return hist

# input image(3 channel)還沒寫


def draw_hist_cdf_3channel(img, dynamic_range, save_path, color_space, tag=None, draw_hist=True, draw_cdf=True, save_img=True, draw_3channel_seperated=False):
    channels = cv2.split(img)
    if color_space == 'YCrCb':
        colors_name = ('Y', 'Cr', 'Cb')
        colors = ('y', 'r', 'b')
    elif color_space == 'BGR':
        colors_name = ('B', 'G', 'R')
        colors = ('b', 'g', 'r')
    elif color_space == 'HSV':
        colors_name = ('H', 'S', 'V')
        colors = ('b', 'r', 'y')

    if draw_3channel_seperated:
        for (ch, con) in zip(channels, colors_name):
            _, _ = draw_hist_cdf(ch, dynamic_range, save_path, con, tag)

    fig = plt.figure()

    for (channel, color_name, color) in zip(channels, colors_name, colors):
        hist, bins = np.histogram(
            channel.flatten(), dynamic_range, [0, dynamic_range])
        plt.bar(range(0, dynamic_range), hist,
                color=color, width=1, alpha=0.5)  # 柱狀圖
        # plt.plot(hist, color = color) #折線圖
        plt.xlim([0, dynamic_range])
        plt.ylim([0, 2000])  # 暫時
        plt.legend((color_name), loc='upper left')
    if save_img:
        fig.savefig(save_path + tag + color_space + "_hist.png", dpi=fig.dpi)
    plt.close()

# input histogram


def draw_hist_cdf2(hist, dynamic_range, save_path, colors_name=None, tag=None, draw_hist=True, draw_cdf=True, save_img=True):
    cdf = hist.cumsum()
    #cdf_normalized = cdf * float(hist.max()) / cdf.max()
    cdf_normalized = cdf / cdf.max() * hist.max()
    fig = plt.figure()
    plt.plot(cdf_normalized, color='b')
    plt.bar(range(0, dynamic_range), hist, color='r', width=1)
    plt.xlim([0, dynamic_range])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    if save_img:
        fig.savefig(save_path + tag + colors_name + "_hist.png", dpi=fig.dpi)
    plt.close()
    return cdf


def create_a_image(hist, h, w):
    created_img = np.zeros((sum(hist), 1)).astype('uint16')
    # print(sum(hist))
    j = 0

    for i in range(len(hist)):
        for k in range(hist[i]):
            created_img[j] = i
            j = j+1

    created_img = np.reshape(created_img, (h, w))

    return created_img

# input image(1 channel)
# def hist_matching(img, target_cdf_img, save_path = None, draw_hist = True):
#    matched = exposure.match_histograms(img, target_cdf_img, channel_axis = None)
#    return matched


def calc_ssim(x, y, dynamic_range):
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    return tf.image.ssim(x, y, (dynamic_range-1))


def tonemapping(img, dynamic_range, save_path, color_space, tag=None, method='Reinhard'):
    # float 32 bits 3 channels
    img = (img/(dynamic_range-1)).astype('float32')  # 0~1
    #print("type(img): ", type(img))
    #print("img.ndim: ", img.ndim)
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


'''
    ###
    #https://learnopencv.com/high-dynamic-range-hdr-imaging-using-opencv-cpp-python/
    
    # Tonemap using Drago's method to obtain 24-bit color image
    # Adaptive Logarithmic Mapping For Displaying High Contrast Scenes
    # bias是偏置函數在 [0, 1] 範圍內的值。0.7 到 0.9 之間的值通常給出最好的結果。默認值為 0.85。
    # 參數是通過反複試驗獲得的。最終輸出乘以 3 只是因為它給出了最令人滿意的結果。
    tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
    ldrDrago = tonemapDrago.process(hdrDebevec)
    ldrDrago = 3 * ldrDrago
    cv2.imwrite("ldr-Drago.jpg", ldrDrago * 255)
    
    # Tonemap using Durand's method obtain 24-bit color image
    # Fast Bilateral Filtering for the Display of High-Dynamic-Range Images
    # 該算法基於將圖像分解為基礎層和細節層。基礎層是使用稱為雙邊濾波器的邊緣保持濾波器獲得的。
    # sigma_space和sigma_color是雙邊濾波器的參數，分別控制空間和顏色域中的平滑量。
    tonemapDurand = cv2.createTonemapDurand(1.5,4,1.0,1,1)
    ldrDurand = tonemapDurand.process(hdrDebevec)
    ldrDurand = 3 * ldrDurand
    cv2.imwrite("ldr-Durand.jpg", ldrDurand * 255)

    # Tonemap using Reinhard's method to obtain 24-bit color image
    #
    # 參數強度應在 [-8, 8] 範圍內。更大的強度值會產生更亮的結果。light_adapt控制光照適應，在 [0, 1] 範圍內。
    # 值 1 表示僅基於像素值的自適應，值 0 表示全局自適應。中間值可用於兩者的加權組合。參數color_adapt控制色度適應，在 [0, 1] 範圍內。
    # 如果該值設置為 1，則通道被獨立處理，如果該值設置為 0，則每個通道的適應級別相同。中間值可用於兩者的加權組合。
    tonemapReinhard = cv2.createTonemapReinhard(1.5, 0,0,0)
    ldrReinhard = tonemapReinhard.process(hdrDebevec)
    cv2.imwrite("ldr-Reinhard.jpg", ldrReinhard * 255)

    # Tonemap using Mantiuk's method to obtain 24-bit color image
    #
    # 參數scale是對比度比例因子。從 0.6 到 0.9 的值產生最佳結果。
    tonemapMantiuk = cv2.createTonemapMantiuk(2.2,0.85, 1.2)
    ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
    ldrMantiuk = 3 * ldrMantiuk
    cv2.imwrite("ldr-Mantiuk.jpg", ldrMantiuk * 255)
    ###
'''
###
'''
#https://oldpan.me/archives/style-transfer-histogram-match
def select_idx(tensor, idx):
    ch = tensor.size(0)
    return tensor.view(-1)[idx.view(-1)].view(ch,-1)

def remap_hist(x, hist_ref):
    ch, n = x.size()
    sorted_x, sort_idx = x.data.sort(1)
    ymin, ymax = x.data.min(1)[0].unsqueeze(1), x.data.max(1)[0].unsqueeze(1)
    hist = hist_ref * n/hist_ref.sum(1).unsqueeze(1)#Normalization between the different lengths of masks.
    cum_ref = hist.cumsum(1)
    cum_prev = torch.cat([torch.zeros(ch,1).cuda(), cum_ref[:,:-1]],1)
    step = (ymax-ymin)/n_bins
    rng = torch.arange(1,n+1).unsqueeze(0).cuda()
    idx = (cum_ref.unsqueeze(1) - rng.unsqueeze(2) < 0).sum(2).long()
    ratio = (rng - select_idx(cum_prev,idx)) / (1e-8 + select_idx(hist,idx))
    ratio = ratio.squeeze().clamp(0,1)
    new_x = ymin + (ratio + idx.float()) * step
    new_x[:,-1] = ymax
    _, remap = sort_idx.sort()
    new_x = select_idx(new_x,idx)
    return new_x
'''
###

###
# https://towardsdatascience.com/a-simple-hdr-implementation-on-opencv-python-2325dbd9c650
# Tonemap HDR image
#tonemap1 = cv2.createTonemap(gamma=2.2)
#res_debevec = tonemap1.process(hdr_debevec.copy())
# Exposure fusion using Mertens
#merge_mertens = cv2.createMergeMertens()
#res_mertens = merge_mertens.process(img_list)
# Convert datatype to 8-bit and save
#res_debevec_8bit = np.clip(res_debevec*255, 0, 255).astype('uint8')
#res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')
#cv2.imwrite(r"C:\Users\felipe.cunha\Documents\venv\HDRTest\ldr_debevec.jpg", res_debevec_8bit)
#cv2.imwrite(r"C:\Users\felipe.cunha\Documents\venv\HDRTest\fusion_mertens.jpg", res_mertens_8bit)
###

'''
IN_hist, bins = np.histogram(IN_img_Y.flatten(),IN_range,[0,IN_range])
GT_hist, bins = np.histogram(GT_img_Y.flatten(),GT_range,[0,GT_range])
IN_cdf = IN_hist.cumsum()
GT_cdf = GT_hist.cumsum()

IN_cdf_normalized = IN_cdf * float(IN_hist.max()) / IN_cdf.max()
fig = plt.figure()
plt.plot(IN_cdf_normalized, color = 'b')
plt.bar(range(0,IN_range) , IN_hist, color = 'r', width=1)
plt.xlim([0,IN_range])
plt.legend(('cdf','histogram'), loc = 'upper left')
fig.savefig("./output_img/IN_hist.png",dpi=fig.dpi)
plt.close()

GT_cdf_normalized = GT_cdf * float(GT_hist.max()) / GT_cdf.max()
fig = plt.figure()
plt.plot(GT_cdf_normalized, color = 'b')
plt.bar(range(0,GT_range) , GT_hist, color = 'r', width=1)
plt.xlim([0,GT_range])
plt.legend(('cdf','histogram'), loc = 'upper left')
fig.savefig("./output_img/IN_hist.png",dpi=fig.dpi)
plt.close()
'''
'''
# 防呆(待修)
for i=1:1:upH:
    for j=1:1:upW:
        if input(i,j)<0:
            input(i,j)=0
        if input(i,j)>65535:
            input(i,j)=65535
            
# 防呆(待修)
for i=1:1:Y_height:
    for j=1:1:Y_width:
        if input(i,j)<0:
            input(i,j)=0
        if input(i,j)>65535:
            input(i,j)=65535
'''
'''
IN_img_Y  = IN_img_YCrCb[:, :, 0]
IN_img_Cr = IN_img_YCrCb[:, :, 1]
IN_img_Cb = IN_img_YCrCb[:, :, 2]
GT_img_Y  = GT_img_YCrCb[:, :, 0]
GT_img_Cr = GT_img_YCrCb[:, :, 1]
GT_img_Cb = GT_img_YCrCb[:, :, 2]
'''
