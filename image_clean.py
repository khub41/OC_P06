from PIL import Image
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os


def show_img(img):
    """Convenience function to display a typical color image"""
    return plt.imshow(cv2.cvtColor(img, cv2.CV_32S))


def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray


def gen_sift_features(gray_img):
    sift = cv2.SIFT_create()
    # kp is the keypoints
    #
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc


def show_sift_features(gray_img, color_img, kp):
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))


data = pd.read_csv('data/Flipkart/flipkart_com-ecommerce_sample_1050.csv')

product = data.sample(1, random_state=41)

img = cv2.imread('data/Flipkart/Images/' + product.image.iloc[0])
# show_img(img)

img_bw = to_gray(img)
# show_img(img_bw)

img_bw_kp, img_bw_desc = gen_sift_features(img_bw)

show_sift_features(img_bw, img, img_bw_kp)




sift = cv2.SIFT_create()
kp = sift.detect(img_bw,None)
img_bw = cv2.drawKeypoints(img_bw, kp, img)




