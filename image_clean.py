from PIL import Image
import pandas as pd
import cv2
data = pd.read_csv('data/Flipkart/flipkart_com-ecommerce_sample_1050.csv')

product = data.sample(1)
# img = Image.open('data/Flipkart/Images/' + product.image.iloc[0])
# # img.show()
img = cv2.imread('data/Flipkart/Images/' + product.image.iloc[0])
img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp = sift.detect(img_bw,None)
img_bw = cv2.drawKeypoints(img_bw, kp, img)
cv2.imwrite(f'data/Images/sift/sift_kp_{product.iloc[0].name}.jpg',img_bw)



