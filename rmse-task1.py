import os
import matplotlib

import numpy as np
import scipy.io as sio
# import imageio
import matplotlib.pyplot as plt


rmse = np.zeros((8,16), dtype='float32')
# ICVL
fns = [
        # 'nachal_0823-1147.mat',
        'negev_0823-1003.mat',
        'tree_0822-0853.mat'
       ]

for fn in fns:
    gt_path = 'E:/Data/icvl/icvl512_101_gt_16band'
    gt = sio.loadmat(os.path.join(gt_path, fn))['gt']

    wb_path = 'E:/周报/JSTSP/对比方法/result/icvl/DM_Traditional/numpy/WB'
    wb = sio.loadmat(os.path.join(wb_path, fn))['pred']

    grmr_path = 'E:/周报/JSTSP/对比方法/result/icvl/DM_Traditional/numpy/GRMR'
    grmr = sio.loadmat(os.path.join(grmr_path, fn))['pred']

    ppid_path = 'E:/周报/JSTSP/对比方法/result/icvl/DM_Traditional/numpy/PPID'
    ppid = sio.loadmat(os.path.join(ppid_path, fn))['pred']

    hsup_path = 'E:/周报/JSTSP/对比方法/result/icvl/hsup/task1'
    hsup = sio.loadmat(os.path.join(hsup_path, fn))['pred'].transpose(1,2,0)

    mdrn_path = 'E:/周报/JSTSP/对比方法/result/icvl/mdrn_16channel_base/icvl/task1'
    mdrn = sio.loadmat(os.path.join(mdrn_path, fn))['pred'].transpose(1,2,0)

    mcan_path = 'E:/周报/JSTSP/对比方法/result/icvl/mcan/task1'
    mcan = sio.loadmat(os.path.join(mcan_path, fn))['pred'].transpose(1,2,0)

    dgmsp_path = 'H:/HSI_DM/majorRevision/harvard/dgsmp_16channel_base/icvl/task1'
    dgmsp = sio.loadmat(os.path.join(dgmsp_path, fn))['pred'].transpose(1,2,0)

    ours_path = 'E:/周报/JSTSP/对比方法/result/icvl/admmn_16channel_alpha/icvl/task1'
    ours = sio.loadmat(os.path.join(ours_path, fn))['pred'].transpose(1,2,0)

    h,w,c = gt.shape
    for k in range(c):
        rmse[0][k] = np.sqrt(np.mean((gt[:,:,k]-wb[:,:,k])**2))*255
        rmse[1][k] = np.sqrt(np.mean((gt[:,:,k]-grmr[:,:,k])**2))*255
        rmse[2][k] = np.sqrt(np.mean((gt[:,:,k]-ppid[:,:,k])**2))*255
        rmse[3][k] = np.sqrt(np.mean((gt[:,:,k]-hsup[:,:,k])**2))*255
        rmse[4][k] = np.sqrt(np.mean((gt[:,:,k]-mdrn[:,:,k])**2))*255
        rmse[5][k] = np.sqrt(np.mean((gt[:,:,k]-mcan[:,:,k])**2))*255
        rmse[5][k] = np.sqrt(np.mean((gt[:,:,k]-mcan[:,:,k])**2))*255
        rmse[6][k] = np.sqrt(np.mean((gt[:,:,k]-dgmsp[:,:,k])**2))*255
        rmse[7][k] = np.sqrt(np.mean((gt[:,:,k]-ours[:,:,k])**2))*255

    x = range(400, 701, 20)
    font = {'size':20, 'weight': 'black'}
    plt.figure(figsize=(6,6))
    plt.plot(x, rmse[0], 'b-', label='WB', linewidth=2.2)
    plt.plot(x, rmse[1], 'g-', label='GRMR', linewidth=2.2)
    plt.plot(x, rmse[2], 'c-', label='PPID', linewidth=2.2)
    plt.plot(x, rmse[3], 'cyan', label='HSUp', linewidth=2.2)
    plt.plot(x, rmse[4], 'm-', label='MDRN', linewidth=2.2)
    plt.plot(x, rmse[5], 'y-', label='MCAN', linewidth=2.2)
    plt.plot(x, rmse[6], 'darkorange', label='DGMSP', linewidth=2.2)
    plt.plot(x, rmse[7], 'r-', label='Ours', linewidth=2.2)
    plt.xlabel('Wavelength [nm]', fontdict=font)
    plt.ylabel('RMSE', fontdict=font)
    ax = plt.gca()
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontweight('black') for label in labels]
    [label.set_fontsize(15) for label in labels]

    plt.xlim(400,700)
    # plt.legend(loc='center', bbox_to_anchor=(0.17,0.72), fontsize=16)
    plt.savefig('E:/周报/JSTSP/对比方法/result/majorRevision/dgmsp/spa/icvl/'+fn[:-4]+'_rmse.pdf')
    plt.show()



'''
rmse = np.zeros((8,16), dtype='float32')
# Harvard
fns = ['imgc7.mat','imge3.mat',
       'imgg2.mat']
# fns = ['imgg2.mat']
for fn in fns:
    gt_path = 'E:/Data/Harvard/test_16channel'
    gt = sio.loadmat(os.path.join(gt_path, fn))['gt']

    wb_path = 'E:/周报/JSTSP/对比方法/result/harvard/DM_Traditional/numpy/WB'
    wb = sio.loadmat(os.path.join(wb_path, fn))['pred']

    grmr_path = 'E:/周报/JSTSP/对比方法/result/harvard/DM_Traditional/numpy/GRMR'
    grmr = sio.loadmat(os.path.join(grmr_path, fn))['pred']

    ppid_path = 'E:/周报/JSTSP/对比方法/result/harvard/DM_Traditional/numpy/PPID'
    ppid = sio.loadmat(os.path.join(ppid_path, fn))['pred']

    hsup_path = 'E:/周报/JSTSP/对比方法/result/harvard/hsup/task1'
    hsup = sio.loadmat(os.path.join(hsup_path, fn))['pred'].transpose(1,2,0)

    mdrn_path = 'E:/周报/JSTSP/对比方法/result/harvard/mdrn_16channel_base/harvard/task1'
    mdrn = sio.loadmat(os.path.join(mdrn_path, fn))['pred'].transpose(1,2,0)

    mcan_path = 'E:/周报/JSTSP/对比方法/result/harvard/mcan/task1'
    mcan = sio.loadmat(os.path.join(mcan_path, fn))['pred'].transpose(1,2,0)

    dgmsp_path = 'H:/HSI_DM/majorRevision/harvard/dgsmp_16channel_base/harvard/task1'
    dgmsp = sio.loadmat(os.path.join(dgmsp_path, fn))['pred'].transpose(1,2,0)

    ours_path = 'E:/周报/JSTSP/对比方法/result/harvard/admmn_16channel_alpha/harvard/task1'
    ours = sio.loadmat(os.path.join(ours_path, fn))['pred'].transpose(1,2,0)

    h,w,c = gt.shape
    for k in range(c):
        rmse[0][k] = np.sqrt(np.mean((gt[:,:,k]-wb[:,:,k])**2))*255
        rmse[1][k] = np.sqrt(np.mean((gt[:,:,k]-grmr[:,:,k])**2))*255
        rmse[2][k] = np.sqrt(np.mean((gt[:,:,k]-ppid[:,:,k])**2))*255
        rmse[3][k] = np.sqrt(np.mean((gt[:,:,k]-hsup[:,:,k])**2))*255
        rmse[4][k] = np.sqrt(np.mean((gt[:,:,k]-mdrn[:,:,k])**2))*255
        rmse[5][k] = np.sqrt(np.mean((gt[:,:,k]-mcan[:,:,k])**2))*255
        rmse[6][k] = np.sqrt(np.mean((gt[:,:,k]-dgmsp[:,:,k])**2))*255
        rmse[7][k] = np.sqrt(np.mean((gt[:,:,k]-ours[:,:,k])**2))*255

    x = range(400, 701, 20)
    font = {'size':20, 'weight': 'black'}
    plt.figure(figsize=(6,6))
    plt.plot(x, rmse[0], 'b-', label='WB', linewidth=2.2)
    plt.plot(x, rmse[1], 'g-', label='GRMR', linewidth=2.2)
    plt.plot(x, rmse[2], 'c-', label='PPID', linewidth=2.2)
    plt.plot(x, rmse[3], 'cyan', label='HSUp', linewidth=2.2)
    plt.plot(x, rmse[4], 'm-', label='MDRN', linewidth=2.2)
    plt.plot(x, rmse[5], 'y-', label='MCAN', linewidth=2.2)
    plt.plot(x, rmse[6], 'darkorange', label='DGMSP', linewidth=2.2)
    plt.plot(x, rmse[7], 'r-', label='Ours', linewidth=2.2)
    plt.xlabel('Wavelength [nm]', fontdict=font)
    plt.ylabel('RMSE', fontdict=font)
    ax = plt.gca()
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontweight('black') for label in labels]
    [label.set_fontsize(15) for label in labels]

    plt.xlim(400,700)
    # plt.legend(loc='center', bbox_to_anchor=(0.16,0.75), fontsize=16)
    plt.savefig('E:/周报/JSTSP/对比方法/result/majorRevision/dgmsp/spa/harvard/'+fn[:-4]+'_rmse.pdf')
    plt.show()
'''