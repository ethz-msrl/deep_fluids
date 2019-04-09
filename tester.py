from __future__ import print_function

import os
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from glob import glob

import matplotlib
# need this to generate plots over terminal
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import GeneratorBE3
from ops import get_conv_shape, show_all_variables, divergence3, jacobian3
from util import prepare_dirs_and_logger

class Tester(object):
    def __init__(self, config):
        """
        Testing based on a separate batch of test data

        Assumes 3D, RE arch
        """
        #prepare_dirs_and_logger(config)
        self.config = config

        assert(config.load_data_path)
        self.root = config.load_data_path        

        self.model_dir = config.load_model_dir
        self.generate_streamplots = config.generate_streamplots
        if self.generate_streamplots:
            self.streamplots_dir = os.path.join(self.model_dir, 'streamplots')
            if not os.path.exists(self.streamplots_dir):
                os.makedirs(self.streamplots_dir)

        self.stats_dir = os.path.join(self.model_dir, 'stats')
        if not os.path.exists(self.stats_dir):
            os.makedirs(self.stats_dir)

        assert(self.model_dir)

        # read data generation arguments
        self.args = {}
        with open(os.path.join(self.root, 'args.txt'), 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                arg, arg_value = line[:-1].split(': ')
                self.args[arg] = arg_value
        self.c_num = int(self.args['num_param'])
        
        assert(config.arch == 'de')
        assert(config.is_3d)

        self.use_c = config.use_curl

        self.res_x = config.res_x
        self.res_y = config.res_y
        self.res_z = config.res_z
        # because we only support 3d velocity fields
        self.depth = 3

        self.num_conv = config.num_conv

        self.test_b_num = 1

        self.filters = config.filters
        self.output_shape = (self.res_x, self.res_y, self.res_z, self.depth)
        self.repeat = config.repeat

        self.data_type = config.data_type
        r = [float(self.args['min_b']), float(self.args['max_b'])]
        self.x_range = max(abs(r[0]), abs(r[1]))

        p_min = float(self.args['min_c'])
        p_max = float(self.args['max_c'])
        p_num = int(self.args['num_param'])
        self.y_range = []
        self.y_num = []
        for i in range(self.c_num):
            p_min = float(self.args['min_c'])
            p_max = float(self.args['max_c'])
            p_num = int(self.args['num_param'])
            self.y_range.append([p_min, p_max])
            self.y_num.append(p_num)

        #self.paths = glob("{}/{}/*".format(self.root, config.test_path))
        self.paths = glob("{}/{}/*".format(self.root, 'v'))
        assert(self.paths)
        self.idx_test = np.loadtxt(os.path.join(self.root, config.test_idx),dtype=np.int)
        self.n_test = len(self.idx_test)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)
        self.sess = tf.Session(config=sess_config)

        self.z = tf.placeholder(dtype=tf.float32, shape=(self.test_b_num, self.c_num))
        self.G_, _ = GeneratorBE3(self.z, self.filters, self.output_shape,
                                 num_conv=self.num_conv, repeat=self.repeat, reuse=False)
        self.saver= tf.train.Saver()
        self.model_fn = tf.train.latest_checkpoint(self.model_dir)
        assert(self.model_fn)
        self.saver.restore(self.sess, self.model_fn) 

        show_all_variables()

    def denorm(self, x=None, y=None):
        # input range [-1, 1] -> original range
        if x is not None:
            x *= self.x_range

        if y is not None:
            r = self.y_range
            for i, ri in enumerate(self.y_range):
                y[i] = (y[i]+1) * 0.5 * (ri[1]-ri[0]) + ri[0]
        return x, y

    def test(self):
        """
        Runs the tests
        """
        ss_res = 0
        ss_tot = 0
        for i, idx in enumerate(tqdm(self.idx_test)):
            path = self.paths[idx] 
            x, y = preprocess(path, self.data_type, self.x_range,
                    self.y_range)
            xd, _ = self.denorm(x.copy())
            # x_div = divergence3_np(xd)
            # print('divergence max: %f mean:% f' % (np.max(x_div), np.mean(x_div)))
            if self.use_c:
                _, G_s = jacobian3(self.G_)
                G_ = self.sess.run(G_s, {self.z: y})  
                
            else:
                G_ = self.sess.run(self.G_, {self.z: y})  
            G_, _ = self.denorm(x=G_)
            G_ = np.squeeze(G_)
            a, b = compute_ss(xd, G_)
            ss_res += a; ss_tot += b
            rmse = np.sqrt(a/np.prod(x.shape[:-1:]))
            r2 = np.ones((xd.shape[-1]), np.float) - a / b 
            # verbose printing
            #print('%d r2: %s, max current: %f' % (i, r2, np.max(np.abs(y))*35))
            np.savetxt(os.path.join(self.stats_dir, '{:03d}.stats'.format(i)), 
                    np.vstack((r2, rmse)))

            if self.generate_streamplots:
                self.streamplot_slice(x, G_, 'xy_{:03d}.png'.format(i))

            if ((i-1) % 20) == 0: 
                r2_t = np.ones((xd.shape[-1]), np.float) - ss_res / ss_tot
                T = i * np.prod(x.shape[:-1:])
                rmse_t = np.sqrt(ss_res / T)
                print('r2: %s' % r2_t)
                print('rmse: %s mT' % (1000*rmse_t))

        save_path = os.path.join(self.model_fn + '.test_stats')
        np.savetxt(save_path, np.vstack((r2_t, rmse_t)))

    def test_single(self, path):
        x, y = preprocess(path, self.data_type, self.x_range,
                self.y_range)
        xd, _ = self.denorm(x.copy())
        G_ = self.sess.run(self.G_, {self.z: y})  
        G_, _ = self.denorm(x=G_)
        G_ = np.squeeze(G_)
        print('saving to %s.test' % path)
        np.savez_compressed(path + '.test', x=G_)

    def streamplot_slice(self, x, x_, filename):
        assert(x.shape == x_.shape)
        c_id = [int(x.shape[0]/2), int(x.shape[2]/2)]
        xy = x[c_id[0], :, :, :]
        xy_ = x_[c_id[0], :, :, :]
        fp = os.path.join(self.streamplots_dir, filename)
        streamplot2(xy, xy_, fp) 

def compute_ss(obs, pred):
    """
    Computes the sum of square residuals and total used for calculating R2 score

    ss_res = sum((pred - obs)^2)
    ss_tot = sum((obs - mean(obs))^2)

    The tensors are reduced on all axes except the last
    For example a tensor of size [8,8,8,3] will result in a 
    vector of length 3

    Args:
        obs: a tensor of observed values
        pred: a tensor the same size as pred of predicted values
    Returns:
        a tuple containting (ss_res, ss_tot) where ss_res and ss_tot are vectors of length corresponding to the last
        dimedimension of obs and pred
    """
    assert(obs.shape == pred.shape)
    # dims to reduce
    d_r = tuple(range(obs.ndim-1))
    ss_res = np.sum(np.square(obs - pred), d_r)
    obs_m = np.mean(obs, axis=d_r)
    ss_tot = np.sum(np.square(obs - obs_m), d_r)
    return ss_res, ss_tot

def compute_r2(obs, pred):
    """
    Computes the goodness of fit R2 score on all axes

    The tensors are reduced on all axes except the last
    For example a tensor of size [8,8,8,3] will result in a 
    vector of length 3

    r2 = 1 - ss_res / ss_tot

    Args:
        obs: a tensor of observed values
        pred: a tensor the same size as pred of predicted values
    Returns:
        a vector of R2 values of length corresponding to the last dimension of obs and pred 
    """
    assert(obs.shape == pred.shape)
    # dims to reduce
    d_r = tuple(range(obs.ndim-1))
    ss_res = np.sum(np.square(obs - pred), d_r)
    obs_m = np.mean(obs, axis=d_r)
    ss_tot = np.sum(np.square(obs - obs_m), d_r)
    return np.ones((obs.shape[obs.ndim-1]), np.float) - ss_res / ss_tot

def preprocess(file_path, data_type, x_range, y_range):    
    with np.load(file_path) as data:
        x = data['x']
        y = data['y']

    # normalize
    if data_type[0] == 'd':
        x = x*2 - 1
    else:
        x /= x_range
        
    for i, ri in enumerate(y_range):
        y[i] = (y[i]-ri[0]) / (ri[1]-ri[0]) * 2 - 1

    # need to make y
    y = y[np.newaxis, :]
    return x, y

def streamplot2(x, x_, filename, density=1.5, scale=50.):
    h, w = x.shape[0], x.shape[1]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(w*0.01*scale,h*0.01*scale)
    # print(fig.get_size_inches(), fig.dpi)
    fig.frameon = False
    streamplot(x, ax1, density, scale)
    streamplot(x_, ax2, density, scale)
    fig.savefig(filename, bbox_inches='tight')
    #fig.clf()
    plt.close(fig)

def streamplot(x, ax, density=2.0, scale=50.0):
    # uv: [y,x,2]
    # print(x.shape)
    u = x[::-1,:,0]
    v = x[::-1,:,1]

    h, w = x.shape[0], x.shape[1]
    y0, y1 = (0,h-1)
    x0, x1 = (0,w-1)
    Y, X = np.ogrid[y0:y1:complex(0,h), x0:x1:complex(0,w)]
    speed = np.sqrt(u*u + v*v)
    lw = 2*speed / speed.max() + 0.5
    # color = speed / speed.max()
    color = 'k'

    ax.set_axis_off()    
    ax.streamplot(X, Y, u, v, color=color, linewidth=lw, # cmap=plt.cm.inferno, 
        density=density, arrowstyle='->', arrowsize=1.0)

    ax.set_aspect('equal')
    ax.figure.subplots_adjust(bottom=0, top=1, left=0, right=1)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

def divergence3_np(x):
    dudx = x[:-1,:-1,1:,0] - x[:-1,:-1,:-1,0]
    dvdy = x[:-1,1:,:-1,1] - x[:-1,:-1,:-1,1]
    dwdz = x[1:,:-1,:-1,2] - x[:-1,:-1,:-1,2]
    return dudx + dvdy + dwdz

if __name__ == '__main__':
    from config import get_config
    config, unparsed = get_config()

    tester = Tester(config)
    #paths = glob("{}/{}/*".format(config.load_data_path, config.test_path))
    #tester.test_single(paths[0])
    tester.test()

