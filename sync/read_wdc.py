import cv2

path = 'F:\\Remote_SR\\dataset\\washingtonDC\\Hyperspectral_Project\\dc.tif'

img = cv2.imread(path, 2)

print(img.shape)


# from libtiff import TIFF
# tif = TIFF.open('filename.tif', mode='r')
# img = tif.read_image()