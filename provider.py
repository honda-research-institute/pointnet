import os
import sys
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

""" 
=== train/test.lst ===
 filename   class
 pos/xxx.pcd  1
 neg/xxx.pcd  0
   ...
"""
def gen_list_files(dataset_dir, train_pct=0.7):
    pos_dir = os.path.join(dataset_dir, 'pos')
    neg_dir = os.path.join(dataset_dir, 'neg')
    if os.path.exists(pos_dir) and os.path.exists(neg_dir):
        all_lines = []
        fnames = os.listdir(pos_dir)
        for fname in fnames:
            all_lines.append('pos/'+fname+' 1')
        fnames = os.listdir(neg_dir)
        for fname in fnames:
            all_lines.append('neg/' + fname + ' 0')

        # Shuffle
        np.random.shuffle(all_lines)

        # Split and write-out
        f_train = open(os.path.join(dataset_dir, 'train.lst'), 'w')
        f_test = open(os.path.join(dataset_dir, 'test.lst'), 'w')
        for line in all_lines:
            if np.random.rand()<=train_pct:
                f_train.write(line+'\n')
            else:
                f_test.write(line+'\n')
        f_train.close()
        f_test.close()
    else:
        sys.exit('Error! Dataset pos and neg not found at specified location')


def getDataFiles(list_filename):
    return [line.rstrip().split(' ') for line in open(list_filename)]


if __name__ == "__main__":
    gen_list_files('/home/jhuang/Kitti/jhuang')
