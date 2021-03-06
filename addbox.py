import cv2
import os


def addbox(img_path, out_path, sbcord, bbcord=None, sbsize=50, bbsize=200, sbthickness=2, bbthickness=1, color=(0,0,255)):
    """ sb = small box, 
        bb = big box, 
        sbcord = upper left cordinate of small box 
        bbcord = upper left cordinate of big box 
    """
    image = cv2.imread(img_path)
    w, h = image.shape[0], image.shape[1]
    
    # big box.
    if bbcord is None:
        bbcord = (w-bbsize-bbthickness, h-bbsize-bbthickness)
    first_point_b = bbcord
    last_point_b = (bbcord[0] + bbsize, bbcord[1])
    cv2.rectangle(image, first_point_b, last_point_b, color, bbthickness)

    # small box.
    first_point_s = sbcord
    last_point_s = (first_point_s[0] + sbsize, first_point_s[1] + sbsize)
    cv2.rectangle(image, first_point_s, last_point_s, color, sbthickness)

    # crop and combine.
    crop_img = image[first_point_s[1]:first_point_s[1]+sbsize, first_point_s[0]:first_point_s[0]+sbsize]
    crop_img = cv2.resize(crop_img, (bbsize, bbsize))
    image[first_point_b[0]:first_point_b[0]+bbsize, first_point_b[1]:first_point_b[1]+bbsize] = crop_img

    cv2.imwrite(out_path, image)


if __name__ == '__main__':
    image_path = '/mnt/e/周报/JSTSP/对比方法/result/harvard/admmn_16channel_alpha/harvard/task1_error'
    fns = os.listdir(image_path)
    fns = ['imgd5.png']
    
    output_path = image_path + '_out'
    os.makedirs(output_path, exist_ok=True)

    for i in range(len(fns)):
        if fns[i][0] == '.': continue
        
        print(fns[i])
        addbox(os.path.join(image_path, fns[i]), 
               os.path.join(output_path, fns[i]),
               sbcord=(60,15))



"""
harvard-task1
'/mnt/e/周报/JSTSP/对比方法/result/harvard/admmn_16channel_alpha/harvard/task1_error'
'/mnt/e/周报/JSTSP/对比方法/result/harvard/admmn_16channel_base/epoch50/task1_error'

fns = ['imgd6.png'] sbcord=(60,430)

'imgc8.png'
sbcord=(270,170),bbcord=(312,0)
last_point_b = (bbcord[0] + bbsize, bbcord[1])

'imgd5.png'
sbcord=(60,15)

"""