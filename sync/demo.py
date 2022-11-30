from addbox_gui import select_roi, addbox
import os


def test_addbox_diff():
    fns = ['flowers_ms.png', 'imge3.png']
    fns = fns[:1]
    root = 'C:\\Users\\liangzy\\Desktop\\result\\paper_png'
    methods = ['a_plus', 'nlrtatv', 'cnmf', 'sspsr', 'ercsr', 'bi3dqrnn', 'drn', 'ours']
    
    for fn in fns:
        for method in methods:
            path = os.path.join(root, method, fn)
            save_path = os.path.join(root, method, fn[:-4]+'_box.png')
    # path = 'C:\\Users\\liangzy\\Desktop\\result\\paper_png\\cnmf\\flowers_ms.png'
            pt = select_roi(path, size=100, preview=addbox)
            addbox(path, pt, save=save_path)

if __name__ == '__main__':
    test_addbox_diff()