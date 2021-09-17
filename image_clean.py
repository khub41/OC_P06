from PIL import Image
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

PATH = 'data/Flipkart/'
DATA_FILE_NAME = ''


def show_img(img):
    """Convenience function to display a typical color image"""
    return plt.imshow(cv2.cvtColor(img, cv2.CV_32S))


def to_gray(color_img):
    """Converts a RGB image into a gray scale image"""
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray


def gen_sift_features(gray_img):
    """Converts an image into keypoints and dscriptors (128 dimensional vectors)"""
    sift = cv2.SIFT_create()

    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc


def show_sift_features(gray_img, color_img, kp):
    """Show the keypoints on the original picture"""
    fig, ax = plt.subplots(1)
    ax.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))
    plt.show()


def get_descriptors(filename, equalize=True, show_pre_process_review=False):
    """From a filename, gets the list of descriptors"""
    img = cv2.imread(filename)
    if equalize:
        dst = equalize_image(img, show=show_pre_process_review)
    else:
        dst = to_gray(img)
    kp, desc = gen_sift_features(dst)
    return desc


def equalize_image(img, show=True):
    src_gray = to_gray(img)
    blur = cv2.GaussianBlur(src_gray, (5, 5), 500)
    dst = cv2.equalizeHist(blur)

    if show:
        fig_im, ax_im = plt.subplots(3, 4)
        ax_im[0][0].imshow(cv2.cvtColor(img, cv2.CV_32S))
        ax_im[0][0].set_title("Original Color picture")
        ax_im[0][1].imshow(src_gray, cmap='gray')
        ax_im[0][2].imshow(blur, cmap='gray')
        ax_im[0][3].imshow(dst, cmap='gray')

        ax_im[1][1].hist(np.concatenate(src_gray.__array__()), density=True, bins=256, cumulative=False)
        ax_im[0][1].set_title("Original Grayscale")
        ax_im[2][1].hist(np.concatenate(src_gray.__array__()), density=True, bins=256, cumulative=True)

        ax_im[1][2].hist(np.concatenate(blur.__array__()), density=True, bins=256, cumulative=False)
        ax_im[0][2].set_title("After Gaussian Blur")
        ax_im[2][2].hist(np.concatenate(blur.__array__()), density=True, bins=256, cumulative=True)

        ax_im[1][3].hist(np.concatenate(dst.__array__()), density=True, bins=256, cumulative=False)
        ax_im[0][3].set_title("After equalization")
        ax_im[2][3].hist(np.concatenate(dst.__array__()), density=True, bins=256, cumulative=True)

        for i in range(1, 4):
            ax_im[1][i].set_title("Grayscale distribution")
            ax_im[2][i].set_title("Cumulative")
        plt.show()

    return dst


def main():
    data = pd.read_csv('data/Flipkart/flipkart_com-ecommerce_sample_1050.csv')

    product = data.sample(1, random_state=41)

    img = cv2.imread('data/Flipkart/Images/' + product.image.iloc[0])
    # show_img(img)

    img_bw = to_gray(img)
    # show_img(img_bw)

    img_bw_kp, img_bw_desc = gen_sift_features(img_bw)

    equalized = equalize_image(img)
    equalized_kp, equalized_desc = gen_sift_features(equalized)

    print(img_bw_kp.__len__())
    print(equalized_kp.__len__())

    show_sift_features(img_bw, img, img_bw_kp)
    show_sift_features(equalized, img, equalized_kp)

if __name__ == '__main__':
    main()
