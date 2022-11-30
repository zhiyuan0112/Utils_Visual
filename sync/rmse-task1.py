import os
import matplotlib

import numpy as np
import scipy.io as sio
# import imageio
import matplotlib.pyplot as plt


rmse = np.zeros((5,31), dtype='float32')
# CAVE

fns = ['flowers_ms.mat']

for fn in fns:
    gt_path = 'F:\\Remote_SR\\result_0528\\bi3dqrnn\\cave\\x4\\noise10_K1Blur'
    gt = sio.loadmat(os.path.join(gt_path, fn))['gt']

    wb_path = 'F:\\Remote_SR\\result_0528\\sspsr\\cave\\x4\\noise10_K1Blur'
    wb = sio.loadmat(os.path.join(wb_path, fn))['pred']

    grmr_path = 'F:\\Remote_SR\\result_0528\\ercsr\\cave\\x4\\noise10_K1Blur'
    grmr = sio.loadmat(os.path.join(grmr_path, fn))['pred']

    ppid_path = 'F:\\Remote_SR\\result_0528\\bi3dqrnn\\cave\\x4\\noise10_K1Blur'
    ppid = sio.loadmat(os.path.join(ppid_path, fn))['pred']

    hsup_path = 'F:\\Remote_SR\\result_0528\\drn\\cave\\noise10_K1'
    hsup = sio.loadmat(os.path.join(hsup_path, fn))['pred']

    ours_path = 'F:\\Remote_SR\\result_0528\\ours\\cave\\x4\\noise10_k1'
    ours = sio.loadmat(os.path.join(ours_path, fn))['pred']

    h,w,c = gt.shape
    for k in range(c):
        rmse[0][k] = np.sqrt(np.mean((gt[:,:,k]-wb[:,:,k])**2))*255
        rmse[1][k] = np.sqrt(np.mean((gt[:,:,k]-grmr[:,:,k])**2))*255
        rmse[2][k] = np.sqrt(np.mean((gt[:,:,k]-ppid[:,:,k])**2))*255
        rmse[3][k] = np.sqrt(np.mean((gt[:,:,k]-hsup[:,:,k])**2))*255
        rmse[4][k] = np.sqrt(np.mean((gt[:,:,k]-ours[:,:,k])**2))*255

    x = range(400, 701, 10)
    font = {'size':20, 'weight': 'black'}
    plt.figure(figsize=(6,6))
    plt.plot(x, rmse[0], 'b-', label='SSPSR', linewidth=2.2)
    plt.plot(x, rmse[1], 'g-', label='ERCSR', linewidth=2.2)
    plt.plot(x, rmse[2], 'c-', label='Bi-3DQRNN', linewidth=2.2)
    plt.plot(x, rmse[3], 'cyan', label='DLGNN', linewidth=2.2)
    plt.plot(x, rmse[4], 'r-', label='Ours', linewidth=2.2)
    plt.xlabel('Wavelength [nm]', fontdict=font)
    plt.ylabel('RMSE', fontdict=font)
    ax = plt.gca()
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontweight('black') for label in labels]
    [label.set_fontsize(15) for label in labels]

    plt.xlim(400,700)
    plt.legend(loc='center', bbox_to_anchor=(0.23,0.82), fontsize=16)
    plt.savefig('C:\\Users\\liangzy\\Desktop\\result\\paper\\'+fn[:-4]+'_rmse.pdf')
    plt.show()




# cave
fns = ['flowers_ms.mat']

for fn in fns:
    gt_path = 'F:\\Remote_SR\\result_0528\\bi3dqrnn\\cave\\x4\\noise10_K1Blur'
    gt = sio.loadmat(os.path.join(gt_path, fn))['gt']

    wb_path = 'F:\\Remote_SR\\result_0528\\sspsr\\cave\\x4\\noise10_K1Blur'
    wb = sio.loadmat(os.path.join(wb_path, fn))['pred']

    grmr_path = 'F:\\Remote_SR\\result_0528\\ercsr\\cave\\x4\\noise10_K1Blur'
    grmr = sio.loadmat(os.path.join(grmr_path, fn))['pred']

    ppid_path = 'F:\\Remote_SR\\result_0528\\sfcsr\\cave\\x4\\noise10_K1Blur'
    ppid = sio.loadmat(os.path.join(ppid_path, fn))['pred']

    hsup_path = 'F:\\Remote_SR\\result_0528\\bi3dqrnn\\cave\\x4\\noise10_K1Blur'
    hsup = sio.loadmat(os.path.join(hsup_path, fn))['pred']

    ours_path = 'F:\\Remote_SR\\result_0528\\ours\\cave\\x4\\noise10_k1'
    ours = sio.loadmat(os.path.join(ours_path, fn))['pred']


# harvard
fns = ['imge3.mat']

for fn in fns:
    gt_path = 'F:\\Remote_SR\\result_0528\\bi3dqrnn\\harvard\\x4\\noise10_K1Blur'
    gt = sio.loadmat(os.path.join(gt_path, fn))['gt']

    wb_path = 'F:\\Remote_SR\\result_0528\\sspsr\\harvard\\x4\\noise10_K1Blur'
    wb = sio.loadmat(os.path.join(wb_path, fn))['pred']

    grmr_path = 'F:\\Remote_SR\\result_0528\\ercsr\\harvard\\x4\\noise10_K1Blur'
    grmr = sio.loadmat(os.path.join(grmr_path, fn))['pred']

    ppid_path = 'F:\\Remote_SR\\result_0528\\drn\\harvard\\x4\\noise10_K1Blur'
    ppid = sio.loadmat(os.path.join(ppid_path, fn))['pred']

    hsup_path = 'F:\\Remote_SR\\result_0528\\bi3dqrnn\\harvard\\x4\\noise10_K1Blur'
    hsup = sio.loadmat(os.path.join(hsup_path, fn))['pred']

    ours_path = 'F:\\Remote_SR\\result_0528\\ours\\harvard\\x4\\noise10_k1'
    ours = sio.loadmat(os.path.join(ours_path, fn))['pred']