import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# fig, axes = plt.subplots(5, 1, figsize=(20, 5))
# fig.subplots_adjust(hspace=8)

# # 第一个colorbar使用线性的Normalize.
# cmap1 = copy.copy(cm.viridis)
# norm1 = mcolors.Normalize(vmin=0, vmax=50)
# im1 = cm.ScalarMappable(norm=norm1, cmap=matplotlib.cm.jet,)
# cbar1 = fig.colorbar(
#     im1, cax=axes[0], orientation='horizontal',
#     # ticks=np.linspace(0, 30, 5),
#     # label='colorbar with Normalize'
# )
# cbar1.ax.tick_params(labelsize=16, labelcolor='black')



fig, axes = plt.subplots(1, 2, figsize=(2, 4))
fig.subplots_adjust(hspace=0, wspace=16)

# 第一个colorbar使用线性的Normalize.
cmap1 = copy.copy(cm.viridis)
norm1 = mcolors.Normalize(vmin=0, vmax=20)
im1 = cm.ScalarMappable(norm=norm1, cmap=matplotlib.cm.jet,)
cbar1 = fig.colorbar(
    im1, cax=axes[0], orientation='vertical',
    # ticks=np.linspace(0, 30, 5),
    # label='colorbar with Normalize'
)
cbar1.ax.tick_params(labelsize=14, labelcolor='black')


plt.savefig('C:\\Users\\liangzy\\Desktop\\result\\paper_bar\\bar_20.pdf')
plt.show()