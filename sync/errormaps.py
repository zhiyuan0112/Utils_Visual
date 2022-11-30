import argparse
import os
import cv2
import matplotlib
# import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


def minmax_normalize(array):
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)


def gen_errormaps(pred_path, gt_path, out_path):
    print(out_path)
    pred = sio.loadmat(pred_path)['pred']
    if 'nlrtatv' in pred_path:
        pred = minmax_normalize(pred)
    gt = sio.loadmat(gt_path)['gt']
    h,w,c = pred.shape
    err = np.zeros((h,w), dtype='float32')
    for i in range(c):
        err += abs(pred[:,:,i] - gt[:,:,i])
    err = err*255/c
    err = err[::-1,:]

    # err = addbox(err, 'debug', sbcord=(60,15))

    # err = np.rot90(np.rot90(np.rot90(err)))

    gci = plt.imshow(err, origin='lower',
                     cmap=matplotlib.cm.jet,
                     norm=matplotlib.colors.Normalize(vmin=0,vmax=20))
    cbar = plt.colorbar(gci, orientation='horizontal')

    plt.show()


    # plt.show()
    atom = pred_path.split('\\')
    # base_path = 'E:\\周报\\RemoteSR\\project_sr\\figs\\result\\cave\\x4_bar30'
    # base_path = 'E:\\周报\\RemoteSR\\project_sr\\figs\\result\\harvard\\x4_bar30'
    # base_path = 'E:\\周报\\RemoteSR\\project_sr\\figs\\result\\wdc\\x4_bar20'
    base_path = 'C:\\Users\\liangzy\\Desktop\\result\\paper'
    # base_path = 'C:\\Users\\liangzy\\Desktop\\result\\paper_png'
    save_path = os.path.join(base_path, atom[3])
    # os.makedirs(save_path, exist_ok=True)
    
    plt.imsave(os.path.join(save_path, atom[-1][:-3]+'pdf'), err, origin='lower',
               cmap=matplotlib.cm.jet, 
               vmin=0,vmax=20)


if __name__ == '__main__':
    pred_path = [
            # 'F:\\Remote_SR\\result_0528\\a_plus\\cave\\x4\\sigma10_k1_x4',
            # 'F:\\Remote_SR\\result_0528\\nlrtatv\\cave\\x4\\sigma10_k1_x4',
            # 'F:\\Remote_SR\\result_0528\\cnmf\\cave\\x4\\noise10_k1',
            # 'F:\\Remote_SR\\result_0528\\sspsr\\cave\\x4\\noise10_K1Blur',
            # 'F:\\Remote_SR\\result_0528\\bi3dqrnn\\cave\\x4\\noise10_K1Blur',
            # 'F:\\Remote_SR\\result_0528\\ercsr\\cave\\x4\\noise10_K1Blur',
            'F:\\Remote_SR\\result_0528\\drn\\cave\\noise10_K1',
            # 'F:\\Remote_SR\\result_0528\\ours\\cave\\x4\\noise10_k1',
        ]
    gt_path = 'F:\\Remote_SR\\result_0528\\bi3dqrnn\\cave\\x4\\noise10_K1Blur'
    fns = ['flowers_ms.mat']
    base_path = 'E:\\周报\\RemoteSR\\project_sr\\figs\\result\\cave\\x4_bar30'

    for setting in pred_path:
        fns = os.listdir(setting)
    
        out_path = setting + '_error_png'
        os.makedirs(out_path, exist_ok=True)

        # fns = ['flowers_ms.mat']
        # fns = ['imge3.mat']

        for fn in fns:
            gen_errormaps(
                os.path.join(setting, fn),
                os.path.join(gt_path, fn),
                os.path.join(out_path, fn)
                )
            # addbox(
            #     'C:\\Users\\liangzy\\Desktop\\result\\paper_png\\a_plus\\test_wdc.png',
            #     'C:\\Users\\liangzy\\Desktop\\result\\paper_png\\a_plus_box\\out.png',
            #     sbcord=(60,15)
            #     )


###################
# CAVE
pred_path = [
        'F:\\Remote_SR\\result_0528\\a_plus\\cave\\x4\\sigma10_k1_x4',
        'F:\\Remote_SR\\result_0528\\nlrtatv\\cave\\x4\\sigma10_k1_x4',
        'F:\\Remote_SR\\result_0528\\cnmf\\cave\\x4\\noise10_k1',
        'F:\\Remote_SR\\result_0528\\sspsr\\cave\\x4\\noise10_K1Blur',
        'F:\\Remote_SR\\result_0528\\bi3dqrnn\\cave\\x4\\noise10_K1Blur',
        'F:\\Remote_SR\\result_0528\\ercsr\\cave\\x4\\noise10_K1Blur',
        'F:\\Remote_SR\\result_0528\\sfcsr\\cave\\x4\\noise10_K1Blur',
        'F:\\Remote_SR\\result_0528\\ours\\cave\\x4\\noise10_k1',
    ]
gt_path = 'F:\\Remote_SR\\result_0528\\bi3dqrnn\\cave\\x4\\noise10_K1Blur'
fns = ['flowers_ms.mat']
base_path = 'E:\\周报\\RemoteSR\\project_sr\\figs\\result\\cave\\x4_bar30'
bar = 30

# PaviaU
pred_path = [
        'F:\\Remote_SR\\result_0528\\a_plus\\paviau\\x4',
        'F:\\Remote_SR\\result_0528\\nlrtatv\\paviau\\x4',
        'F:\\Remote_SR\\result_0528\\cnmf\\paviau\\x4',
        'F:\\Remote_SR\\result_0528\\bi3dqrnn\\paviau\\x4',
        'F:\\Remote_SR\\result_0528\\ercsr\\paviau\\x4',
        'F:\\Remote_SR\\result_0528\\sfcsr\\paviau\\x4',
        'F:\\Remote_SR\\result_0528\\ours\\paviau\\x4',
        'F:\\Remote_SR\\result_0528\\drn\\paviau\\noise10_K1Blur',
    ]
gt_path = 'F:\\Remote_SR\\result_0528\\bi3dqrnn\\paviau\\x4'
bar = 50


# harvard
pred_path = [
        'F:\\Remote_SR\\result_0528\\a_plus\\harvard\\x4\\sigma10_k1_x4',
        'F:\\Remote_SR\\result_0528\\nlrtatv\\harvard\\x4\\sigma10_k1_x4',
        'F:\\Remote_SR\\result_0528\\cnmf\\harvard\\x4\\noise10_k1',
        'F:\\Remote_SR\\result_0528\\sspsr\\harvard\\x4\\noise10_K1Blur',
        'F:\\Remote_SR\\result_0528\\bi3dqrnn\\harvard\\x4\\noise10_K1Blur',
        'F:\\Remote_SR\\result_0528\\ercsr\\harvard\\x4\\noise10_K1Blur',
        'F:\\Remote_SR\\result_0528\\sfcsr\\harvard\\x4\\noise10_K1Blur',
        'F:\\Remote_SR\\result_0528\\ours\\harvard\\x4\\noise10_k1',
    ]
gt_path = 'F:\\Remote_SR\\result_0528\\bi3dqrnn\\harvard\\x4\\noise10_K1Blur'
fns = ['imge3.mat']
base_path = 'E:\\周报\\RemoteSR\\project_sr\\figs\\result\\harvard\\x4_bar30'
bar = 30


# wdc
pred_path = [
        'F:\\Remote_SR\\result_0528\\a_plus\\wdc-x4',
        'F:\\Remote_SR\\result_0528\\nlrtatv\\wdc-x4',
        # 'F:\\Remote_SR\\result_0528\\cnmf\\wdc-x4',
        'F:\\Remote_SR\\result_0528\\sspsr\\wdc-x4',
        'F:\\Remote_SR\\result_0528\\bi3dqrnn\\wdc-x4',
        'F:\\Remote_SR\\result_0528\\bi3dqrnn\\wdc-x4_simple',
        'F:\\Remote_SR\\result_0528\\ercsr\\wdc-x4',
        'F:\\Remote_SR\\result_0528\\sfcsr\\wdc-x4',
        'F:\\Remote_SR\\result_0528\\ours\\wdc-x4',
        'F:\\Remote_SR\\result_0528\\drn\\wdc\\noise10_K1Blur',
    ]
gt_path = 'F:\\Remote_SR\\result_0528\\bi3dqrnn\\wdc-x4'
bar = 20
