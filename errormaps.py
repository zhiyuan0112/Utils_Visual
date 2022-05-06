import argparse
import os

import matplotlib
# import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


def gen_errormaps(pred_path, gt_path, out_path):
    print(out_path)
    pred = sio.loadmat(pred_path)['pred']
    if 'Traditional' not in out_path:
        pred = pred.transpose(1,2,0)
    gt = sio.loadmat(gt_path)['gt']
    h,w,c = pred.shape
    err = np.zeros((h,w), dtype='float32')
    for i in range(c):
        err += abs(pred[:,:,i] - gt[:,:,i])
    err = err*255/c
    err = err[::-1,:]
    if 'Harvard' in gt_path:
        err = err
    else:
        err = np.rot90(np.rot90(np.rot90(err)))

    gci = plt.imshow(err, origin='lower',
                     cmap=matplotlib.cm.jet,
                     norm=matplotlib.colors.Normalize(vmin=0,vmax=5))
    # cbar = plt.colorbar(gci, orientation='horizontal')

    # plt.savefig(out_path[:-4]+'bar.pdf')

    # plt.show()
    
    plt.imsave(out_path[:-3]+'pdf', err, origin='lower',
               cmap=matplotlib.cm.jet, 
               vmin=0,vmax=5)



if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Hyperspectral Image Demosaicking.')
    # parser.add_argument('--pred', type=str, default='', help='pred path.')
    # parser.add_argument('--gt', type=str, default='', help='pred path.')
    # opt = parser.parse_args()

    pred_path = 'H:/HSI_DM/majorRevision/harvard_admmblock/admmn_base_7'
    gt_path = 'E:/Data/Harvard/test'
    fns = os.listdir(pred_path)
   
    out_path = pred_path + '_error_pdf'
    os.makedirs(out_path, exist_ok=True)

    for i in range(len(fns)):
        print(fns[i])
        gen_errormaps(os.path.join(pred_path, fns[i]),
                      os.path.join(gt_path, fns[i]),
                      os.path.join(out_path, fns[i]))


###################
# ICVL
gt_path = 'E:/Data/icvl/icvl512_101_gt_16band'
'E:/周报/JSTSP/对比方法/result/icvl/mdrn_16channel_base/icvl/task1'
'E:/周报/JSTSP/对比方法/result/icvl/mcan/task1'
'E:/周报/JSTSP/对比方法/result/icvl/hsup/task1'
'E:/周报/JSTSP/对比方法/result/icvl/DM_Traditional/numpy/WB'
'E:/周报/JSTSP/对比方法/result/icvl/DM_Traditional/numpy/GRMR'
'E:/周报/JSTSP/对比方法/result/icvl/DM_Traditional/numpy/PPID'
'E:/周报/JSTSP/对比方法/result/icvl/admmn_16channel_base/icvl/task1'
'E:/周报/JSTSP/对比方法/result/icvl/admmn_16channel_alpha/icvl/task1'
# ablation
'E:/周报/JSTSP/对比方法/result/icvl/admmn_16channel_base_7/icvl/task1'
# major revision
'H:/HSI_DM/majorRevision/harvard/dgsmp_16channel_base/icvl/task1'

gt_path = 'E:/Data/icvl/icvl512_101_gt'
'E:/周报/JSTSP/对比方法/result/icvl/mdrn_base/icvl/task2'
'E:/周报/JSTSP/对比方法/result/icvl/mcan/task2'
'E:/周报/JSTSP/对比方法/result/icvl/hsup/task2'
'E:/周报/JSTSP/对比方法/result/icvl/hsrnet_ssr/hsrnet-task2'     'E:/周报/JSTSP/对比方法/result/icvl/hsrnet_ssr/hsrnet-task2/new/epoch25'
'E:/周报/JSTSP/对比方法/result/icvl/fmnet_ssr/fmnet-task2'       'E:/周报/JSTSP/对比方法/result/icvl/fmnet_ssr/fmnet-task2/MSE/epoch3'
'E:/周报/JSTSP/对比方法/result/icvl/admmn_spectrum/icvl/task3'
'E:/周报/JSTSP/对比方法/result/icvl/admmn_base/icvl/task2'
'E:/周报/JSTSP/对比方法/result/icvl/admmn_alpha/icvl/task2'
# major revision
'H:/HSI_DM/majorRevision/harvard/dgsmp_base/icvl/task2'

# Harvard
gt_path = 'E:/Data/Harvard/test_16channel'
'E:/周报/JSTSP/对比方法/result/harvard/mdrn_16channel_base/harvard/task1'
'E:/周报/JSTSP/对比方法/result/harvard/mcan/task1'
'E:/周报/JSTSP/对比方法/result/harvard/hsup/task1'
'E:/周报/JSTSP/对比方法/result/harvard/DM_Traditional/numpy/WB'
'E:/周报/JSTSP/对比方法/result/harvard/DM_Traditional/numpy/GRMR'
'E:/周报/JSTSP/对比方法/result/harvard/DM_Traditional/numpy/PPID'
'E:/周报/JSTSP/对比方法/result/harvard/admmn_16channel_base/epoch50/task1'
'E:/周报/JSTSP/对比方法/result/harvard/admmn_16channel_alpha/harvard/task1'
# major revision
'H:/HSI_DM/majorRevision/harvard/dgsmp_16channel_base/harvard/task1'
# ablation
'H:/HSI_DM/majorRevision/harvard_admmblock/admmn_16channel_base_7'

gt_path = 'E:/Data/Harvard/test'
'E:/周报/JSTSP/对比方法/result/harvard/mdrn_base/harvard/task2'
'E:/周报/JSTSP/对比方法/result/harvard/mcan/task2'
'E:/周报/JSTSP/对比方法/result/harvard/hsup/task2'
'E:/周报/JSTSP/对比方法/result/harvard/hsrnet-task2'  'E:/周报/JSTSP/对比方法/result/harvard/hsrnet-task2/new/epoch5'
'E:/周报/JSTSP/对比方法/result/harvard/fmnet-task2'
'E:/周报/JSTSP/对比方法/result/harvard/admmn_spectrum/harvard/task3'
'E:/周报/JSTSP/对比方法/result/harvard/admmn_base/harvard/task2'
'E:/周报/JSTSP/对比方法/result/harvard/admmn_alpha/harvard/task2'
# major revision
'H:/HSI_DM/majorRevision/harvard/dgsmp_base/harvard/task2'
# ablation


# Major revision

