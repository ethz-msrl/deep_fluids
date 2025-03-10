import os
import numpy as np
from scipy.interpolate import Rbf
from matrix_rbf import DivFreeRBF
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import trange
from util import create_test_train_indices

# this is the kernel type for the regular 3D RBF only. 
# for the div-free RBF the kernel is specified in the initialization of that interpolant
rbf_kernel = 'multiquadric'
use_div_free = False

cmag_dir = os.path.join('data', 'cmag_dataset')
cmag_path = os.path.join(cmag_dir, 'master_feature_matrix_v5.npy')
data = np.load(cmag_path)
nrow = 119
n = data.shape[0] // nrow 
bmin = np.amin(data[:,-3:])
bmax = np.amax(data[:,-3:])

print('data shape', data.shape)
print('# pairs', n)

# remove 10 % for testing
idx_train, idx_test = create_test_train_indices(n, test_size=0.1,shuffle=True,seed=123)

print('training with %d samples testing with %d' % (len(idx_train), len(idx_test)))
np.savetxt(os.path.join(cmag_dir,'idx_train.txt'), idx_train, fmt='%d')
np.savetxt(os.path.join(cmag_dir,'idx_test.txt'), idx_test, fmt='%d')

# # visualize
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(nrow):
#     ax.scatter(data[i,0], data[i,1], data[i,2])
#     ax.quiver(data[i,0], data[i,1], data[i,2], \
#               data[i,-3], data[i,-2], data[i,-1], length=0.3)
#     ax.text(data[i,0], data[i,1], data[i,2], '%d' % i)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()

sampling = True
res = 16
args_file = os.path.join(cmag_dir, 'args.txt')
f = open(args_file, 'w')
f.write('num_param: 8\n')
f.write('min_c: -35\n')
f.write('max_c: 35\n')
f.write('min_b: %f\n' % bmin)
f.write('max_b: %f\n' % bmax)
f.write('res: %d\n' % res)
f.write('sampling: %d\n' % sampling)

train_dir = os.path.join(cmag_dir, 'v')

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
# if not os.path.exists(test_dir):
    # os.makedirs(test_dir)

# for regular grid interpolation
if sampling:
    bbox = [
        np.amin(data[:nrow,0]),
        np.amax(data[:nrow,0]),
        np.amin(data[:nrow,1]),
        np.amax(data[:nrow,1]),
        np.amin(data[:nrow,2]),
        np.amax(data[:nrow,2]),
    ]
    f.write('bbox: {}'.format(bbox))
    px = np.linspace(bbox[0], bbox[1], res) 
    py = np.linspace(bbox[2], bbox[3], res) 
    pz = np.linspace(bbox[4], bbox[5], res) 
    pz_, py_, px_ = np.meshgrid(pz, py, px, indexing='ij')

f.close()

for i in trange(n):
    y = data[i*nrow,3:-3]

    p = data[i*nrow:(i+1)*nrow, :3]
    b = data[i*nrow:(i+1)*nrow, -3:]

    # for interpolation on missing points
    if use_div_free:
        rbf = DivFreeRBF(p, b, kernel='multiquadric', eps=1., normalize=True)
        p1 = p[62,:3] + p[62,:3] - p[38,:3]
        b1 = rbf.evaluate_single(p1)

        p = np.insert(p, 86, p1, axis=0)
        b = np.insert(b, 86, b1, axis=0)

        for j in range(97,0,-24):
            p1 = p[j,:3] + p[j,:3] - p[j-1,:3]
            b1 = rbf.evaluate_single(p1)
            p = np.insert(p, j+1, p1, axis=0)
            b = np.insert(b, j+1, b1, axis=0)
    else:
        bx = Rbf(p[:,0], p[:,1], p[:,2], b[:,0], function=rbf_kernel)
        by = Rbf(p[:,0], p[:,1], p[:,2], b[:,1], function=rbf_kernel)
        bz = Rbf(p[:,0], p[:,1], p[:,2], b[:,2], function=rbf_kernel)

        p1 = p[62,:3] + p[62,:3] - p[38,:3]
        b1 = [bx(p1[0], p1[1], p1[2]), by(p1[0], p1[1], p1[2]), bz(p1[0], p1[1], p1[2])]
        p = np.insert(p, 86, p1, axis=0)
        b = np.insert(b, 86, b1, axis=0)

        for j in range(97,0,-24):
            p1 = p[j,:3] + p[j,:3] - p[j-1,:3]
            b1 = [bx(p1[0], p1[1], p1[2]), by(p1[0], p1[1], p1[2]), bz(p1[0], p1[1], p1[2])]
            p = np.insert(p, j+1, p1, axis=0)
            b = np.insert(b, j+1, b1, axis=0)

    # debug
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for j in range(125):
    #     ax.scatter(p[j,0], p[j,1], p[j,2])
    #     ax.quiver(p[j,0], p[j,1], p[j,2], \
    #               b[j,0], b[j,1], b[j,2], length=0.3)
    #     ax.text(p[j,0], p[j,1], p[j,2], '%d' % j)
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()

    if not sampling:
        x = b.reshape([5,5,5,3])
    else:    
        if use_div_free:
            rbf = DivFreeRBF(p, b, kernel='multiquadric', eps=1., normalize=True)
            pxf_ = px_.flatten(); pyf_ = py_.flatten(); pzf_ = pz_.flatten()
            p_ = np.stack((pxf_, pyf_, pzf_), axis=-1)
            x = rbf.evaluate(p_)
            x = x.reshape((res, res, res, 3))
        else:
            bx = Rbf(p[:,0], p[:,1], p[:,2], b[:,0], function=rbf_kernel)
            by = Rbf(p[:,0], p[:,1], p[:,2], b[:,1], function=rbf_kernel)
            bz = Rbf(p[:,0], p[:,1], p[:,2], b[:,2], function=rbf_kernel)

            bx_ = bx(px_, py_, pz_)
            by_ = by(px_, py_, pz_)
            bz_ = bz(px_, py_, pz_)

            x = np.stack((bx_, by_, bz_), axis=-1) 
        # debug
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(px_, py_, pz_)
        # #import pdb; pdb.set_trace()
        # ax.quiver(px_, py_, pz_, \
        #             x[:,:,:,0], x[:,:,:,1], x[:,:,:,2], length=0.3)
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()
    
    file_path = os.path.join(train_dir, '%04d.npz' % i)
    # if i < n_train:
        # file_path = os.path.join(train_dir, '%04d.npz' % i)
    # else:
        # file_path = os.path.join(test_dir, '%04d.npz' % i)

    np.savez_compressed(file_path, x=x, y=y)
