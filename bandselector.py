import os

import numpy as np
import scipy.io as sio
import imageio


def band_selector(img_path, out_path, band):
    print(img_path, out_path)
    img = sio.loadmat(img_path)['gt']
    print(img.shape)
    output = img[:,:,band].clip(0,1)
    # output = np.flipud(np.rot90(np.fliplr(output)))
    output = (output*255).astype('uint8')
    print(np.max(output), np.min(output))
    imageio.imwrite(out_path[:-3]+'png', output)
    # plt.imsave(out_path[:-3]+'png', output, cmap='gray')
    # plt.imshow(output, cmap='gray')


if __name__ == '__main__':
    # img_path = '/mnt/e/周报/JSTSP/对比方法/result/icvl/DM_Traditional/numpy/WB'
    img_path = '/mnt/e/Data/icvl/icvl512_101_gt'
    # img_path = '/mnt/e/Data/Harvard/test'
    fns = os.listdir(img_path)
    fns = ['prk_0328-1031.mat', 'rmt_0328-1249-1.mat', 'omer_0331-1055.mat']

    band = 30    
    out_path = img_path + '_' + str(band)
    os.makedirs(out_path, exist_ok=True)

    for i in range(len(fns)):
        print(fns[i])
        band_selector(os.path.join(img_path, fns[i]), 
                      os.path.join(out_path, fns[i]),
                      band)