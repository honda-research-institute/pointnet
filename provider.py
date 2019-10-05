import os
import sys
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

pt_cnt_min = 3

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

# ndim = 3 or 4 (with intensity)
def read_bin(fname, ndim = 3):
    """Reads in PointCloud from a bin file.
        Returns: xyz (Nx3)
        None : if file is not found
    """

    if os.path.exists(fname):
        with open(fname, 'rb') as fid:
            data_array = np.fromfile(fid) # Nx3
            xyz = data_array.reshape(-1, ndim)
            return xyz
    else:
        return None

# xyz is Nx3, step = vox_size
# return Kx6 where K is top_n
def compute_ndt(xyz, step, top_n):
    n_pts_total = xyz.shape[0]

    xyz = xyz.transpose() # convert to 3xN
    x_min, x_max = np.min(xyz[0, :]), np.max(xyz[0, :])
    y_min, y_max = np.min(xyz[1, :]), np.max(xyz[1, :])
    z_min, z_max = np.min(xyz[2, :]), np.max(xyz[2, :])
    x_max += 0.1  # add a bit margin so that last point is included
    y_max += 0.1
    z_max += 0.1

    step_x = step_y = step_z = step
    ndt = np.empty((0,7)) # Nx7, (n_pts, mean_x, mean_y, mean_z, cov_x, cov_y, cov_z)
    for x1 in np.arange(x_min, x_max, step_x):
        for y1 in np.arange(y_min, y_max, step_y):
            for z1 in np.arange(z_min, z_max, step_z):
                cond_x = np.logical_and(xyz[0, :] >= x1, xyz[0, :] < x1 + step_x)
                cond_y = np.logical_and(xyz[1, :] >= y1, xyz[1, :] < y1 + step_y)
                cond_z = np.logical_and(xyz[2, :] >= z1, xyz[2, :] < z1 + step_z)
                cond = np.logical_and(cond_x, cond_y)
                cond = np.logical_and(cond, cond_z)
                xyz_sel = xyz[:, cond]
                if xyz_sel.size > 0 and xyz_sel.shape[1] >= pt_cnt_min:
                    xyz_mean = np.mean(xyz_sel, axis=1)
                    xyz_cov = np.cov(xyz_sel)
                    xyz_cov_diag = np.diagonal(xyz_cov)
                    n_pts = xyz_sel.shape[1]
                    this_ndt = np.concatenate(([n_pts], xyz_mean, xyz_cov_diag))
                else:
                    if xyz_sel.size == 0:
                        n_pts = 0
                    else:
                        n_pts = xyz_sel.shape[1]
                    this_ndt = np.concatenate(([n_pts], np.zero(6)))
                ndt = np.vstack((ndt, this_ndt))

    # sort ndt by first column (n_pts) in descending order
    ndt = ndt[ndt[:,0].argsort()[::-1], :]

    if ndt.shape[0]>=top_n:
        return ndt[:top_n, 1:]
    else:
        ndt = np.vstack((ndt, np.zeros((top_n-ndt.shape[0], 7))))
        return ndt[:, 1:]

# Returns ndt (Nx6) and label (int)
def loadDataFile(data_tup, vox_size, top_n):
    bin_file, label = data_tup
    xyz = read_bin(bin_file)
    ndt = compute_ndt(xyz, vox_size, top_n)
    return ndt, int(label)

# Returns [[filename_0, label_0], [filename_1, label_1], ...]
def getDataFiles(list_filename):
    return [line.rstrip().split(' ') for line in open(list_filename)]


if __name__ == "__main__":
    gen_list_files('/home/jhuang/Kitti/jhuang')
