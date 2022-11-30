import cv2
from cv2 import imwrite

img = cv2.imread('F:\\Remote_SR\\dataset\\harvard\\visual\\imge4.png', 0)
img_RGB = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
print(img_RGB.shape)

imwrite('test.png', img_RGB)