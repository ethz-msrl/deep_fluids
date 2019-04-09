import os
import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import trange

cmag_dir = os.path.join('data', 'cmag_dataset_ns')
cmag_path = os.path.join(cmag_dir, 'master_feature_matrix_v3.npy')
data = np.load(cmag_path)
nrow = 119
n = data.shape[0] // nrow 
bmin = np.amin(data[:,-3:])
bmax = np.amax(data[:,-3:])

print('data shape', data.shape)
print('# pairs', n)

# remove 10 % for testing
n_test = int(0.1 * n)
n_train = n - n_test

print('training with %d samples testing with %d' % (n_train, n_test))

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
with open(args_file, 'w') as f:
    f.write('num_param: 8\n')
    f.write('min_c: -35\n')
    f.write('max_c: 35\n')
    f.write('min_b: %f\n' % bmin)
    f.write('max_b: %f\n' % bmax)
    f.write('res: %d\n' % res)
    f.write('sampling: %d\n' % sampling)

train_dir = os.path.join(cmag_dir, 'v')
test_dir = os.path.join(cmag_dir, 'test')

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

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
    px = np.linspace(bbox[0], bbox[1], res) 
    py = np.linspace(bbox[2], bbox[3], res) 
    pz = np.linspace(bbox[4], bbox[5], res) 
    pz_, py_, px_ = np.meshgrid(pz, py, px, indexing='ij')

for i in trange(n):
    y = data[i*nrow,3:-3]

    p = data[i*nrow:(i+1)*nrow, :3]
    b = data[i*nrow:(i+1)*nrow, -3:]

    # for interpolation on missing points
    bx = Rbf(p[:,0], p[:,1], p[:,2], b[:,0])
    by = Rbf(p[:,0], p[:,1], p[:,2], b[:,1])
    bz = Rbf(p[:,0], p[:,1], p[:,2], b[:,2])

    p1 = p[62,:3] + p[62,:3] - p[38,:3]
    b1 = [bx(p1[0], p1[1], p1[2]), by(p1[0], p1[1], p1[2]), bz(p1[0], p1[1], p1[2])]
    p = np.insert(p, 86, p1, axis=0)
    b = np.insert(b, 86, b1, axis=0)

    for j in range(97,0,-24):
        p1 = p[j,:3] + p[j,:3] - p[j-1,:3]
        b1 = [bx(p1[0], p1[1], p1[2]), by(p1[0], p1[1], p1[2]), bz(p1[0], p1[1], p1[2])]
        p = np.insert(p, j+1, p1, axis=0)
        b = np.insert(b, j+1, b1, axis=0)

    # # debug
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
        bx = Rbf(p[:,0], p[:,1], p[:,2], b[:,0])
        by = Rbf(p[:,0], p[:,1], p[:,2], b[:,1])
        bz = Rbf(p[:,0], p[:,1], p[:,2], b[:,2])

        bx_ = bx(px_, py_, pz_)
        by_ = by(px_, py_, pz_)
        bz_ = bz(px_, py_, pz_)

        # # debug
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(px_, py_, pz_)
        # ax.quiver(px_, py_, pz_, \
        #           bx_, by_, bz_, length=0.3)
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()

        x = np.stack((bx_, by_, bz_), axis=-1) 
    
    if i < n_train:
        file_path = os.path.join(train_dir, '%04d.npz' % i)
    else:
        file_path = os.path.join(test_dir, '%04d.npz' % i)

    np.savez_compressed(file_path, x=x, y=y)
