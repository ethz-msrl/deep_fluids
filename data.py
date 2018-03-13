import os
from glob import glob

import threading
import multiprocessing
import signal
import sys
from datetime import datetime

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from utils import streamplot, vortplot

from ops import *

class BatchManager(object):
    def __init__(self, config):
        self.rng = np.random.RandomState(config.random_seed)
        self.root = config.data_path        

        # read data generation arguments
        self.args = {}
        with open(os.path.join(self.root, 'args.txt'), 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                arg, arg_value = line[:-1].split(': ')
                self.args[arg] = arg_value

        self.is_3d = config.is_3d
        self.paths = sorted(glob("{}/{}/*".format(self.root, config.data_type[0])))
        
        self.num_samples = len(self.paths)
        assert(self.num_samples > 0)
        self.batch_size = config.batch_size
        self.epochs_per_step = self.batch_size / float(self.num_samples) # per epoch

        self.data_type = config.data_type
        if self.data_type == 'velocity':
            if self.is_3d: depth = 3
            else: depth = 2
        else:
            depth = 1
        
        self.res_x = config.res_x
        self.res_y = config.res_y
        self.res_z = config.res_z
        self.depth = depth
        self.c_num = int(self.args['num_param'])

        if self.is_3d:
            feature_dim = [self.res_z, self.res_y, self.res_x, self.depth]
        else:
            feature_dim = [self.res_y, self.res_x, self.depth]
        label_dim = [self.c_num]

        min_after_dequeue = 5000
        capacity = min_after_dequeue + 3 * self.batch_size
        self.q = tf.FIFOQueue(capacity, [tf.float32, tf.float32], [feature_dim, label_dim])
        self.x = tf.placeholder(dtype=tf.float32, shape=feature_dim)
        self.y = tf.placeholder(dtype=tf.float32, shape=label_dim)
        self.enqueue = self.q.enqueue([self.x, self.y])
        self.num_threads = np.amin([config.num_worker, multiprocessing.cpu_count(), self.batch_size])

        r = np.loadtxt(os.path.join(self.root, self.data_type[0]+'_range.txt'))
        self.x_range = max(abs(r[0]), abs(r[1]))
        self.y_range = []
        self.y_num = []
        for i in range(self.c_num):
            p_name = self.args['p%d' % i]
            p_min = float(self.args['min_{}'.format(p_name)])
            p_max = float(self.args['max_{}'.format(p_name)])
            p_num = int(self.args['num_{}'.format(p_name)])
            self.y_range.append([p_min, p_max])
            self.y_num.append(p_num)

        vr = np.loadtxt(os.path.join(self.root, 'v_range.txt'))
        self.v_range = max(abs(vr[0]), abs(vr[1]))
        self.to_v_ratio = self.x_range / self.v_range

    def __del__(self):
        try:
            self.stop_thread()
        except AttributeError:
            pass

    def start_thread(self, sess):
        print('%s: start to enque with %d threads' % (datetime.now(), self.num_threads))

        # Main thread: create a coordinator.
        self.sess = sess
        self.coord = tf.train.Coordinator()

        # Create a method for loading and enqueuing
        def load_n_enqueue(sess, enqueue, coord, paths, rng,
                           x, y, data_type, x_range, y_range):
            with coord.stop_on_exception():                
                while not coord.should_stop():
                    id = rng.randint(len(paths))
                    x_, y_ = preprocess(paths[id], data_type, x_range, y_range)
                    sess.run(enqueue, feed_dict={x: x_, y: y_})

        # Create threads that enqueue
        self.threads = [threading.Thread(target=load_n_enqueue, 
                                          args=(self.sess, 
                                                self.enqueue,
                                                self.coord,
                                                self.paths,
                                                self.rng,
                                                self.x,
                                                self.y,
                                                self.data_type,
                                                self.x_range,
                                                self.y_range)
                                          ) for i in range(self.num_threads)]

        # define signal handler
        def signal_handler(signum, frame):
            #print "stop training, save checkpoint..."
            #saver.save(sess, "./checkpoints/VDSR_norm_clip_epoch_%03d.ckpt" % epoch ,global_step=global_step)
            print('%s: canceled by SIGINT' % datetime.now())
            self.coord.request_stop()
            self.sess.run(self.q.close(cancel_pending_enqueues=True))
            self.coord.join(self.threads)
            sys.exit(1)
        signal.signal(signal.SIGINT, signal_handler)

        # Start the threads and wait for all of them to stop.
        for t in self.threads:
            t.start()

    def stop_thread(self):
        # dirty way to bypass graph finilization error
        g = tf.get_default_graph()
        g._finalized = False

        self.coord.request_stop()
        self.sess.run(self.q.close(cancel_pending_enqueues=True))
        self.coord.join(self.threads)

    def batch(self):
        return self.q.dequeue_many(self.batch_size)

    def sample(self, num):
        idx = self.rng.choice(len(self.paths), num).tolist()
        return [self.paths[i] for i in idx]

    def norm(self, x=None, y=None):
        # original range -> output range [-1, 1]
        if x is not None:
            if self.data_type[0] == 'd':
                x = x*2 - 1
            else:    
                x /= self.x_range
            # x = (x-r[0]) / (r[1]-r[0]) * 2 - 1 # rescaling
        
        if y is not None:
            for i, ri in enumerate(self.y_range):
                y[:,i] = (y[:,i]-ri[0]) / (ri[1]-ri[0]) * 2 - 1

        return x, y

    def denorm(self, x=None, y=None):
        # input range [-1, 1] -> original range
        if x is not None:
            if self.data_type[0] == 'd':
                x = (x+1) * 0.5
            else:
                x *= self.x_range
            # x = (x+1) * 0.5 * (r[1]-r[0]) + r[0] # [r0, r1]

        if y is not None:
            r = self.y_range
            for i, ri in enumerate(self.y_range):
                y[:,i] = (y[:,i]+1) * 0.5 * (ri[1]-ri[0]) + ri[0]
        return x, y

    def to_vel(self, x):
        return x*self.to_v_ratio

    def denorm_vel(self, x):
        x *= self.v_range
        # x = (x+1) * 0.5 * (r[1]-r[0]) + r[0] # [r0, r1]
        return x

    def list_from_p(self, p_list):
        path_format = os.path.join(self.root, self.data_type[0], self.args['path_format'])
        filelist = []
        for p in p_list:
            filelist.append(path_format % tuple(p))
        return filelist

    def random_list2d(self, num):
        xs = []
        pis = []
        zis = []
        for _ in range(num):
            pi = []
            for y_max in self.y_num:
                pi.append(self.rng.randint(y_max))

            filepath = self.list_from_p([pi])[0]
            x, y = preprocess(filepath, self.data_type, self.x_range, self.y_range)
            if self.data_type[0] == 'v':
                b_ch = np.zeros((self.res_y, self.res_x, 1))
                x = np.concatenate((x, b_ch), axis=-1)
            elif self.data_type[0] == 'l':
                offset = 0.5
                eps = 1e-3
                x[x<(offset+eps)] = -1
                x[x>-1] = 1
            x = np.clip((x+1)*127.5, 0, 255)
            zi = [(p/float(self.y_num[i]-1))*2-1 for i, p in enumerate(pi)] # [-1,1]

            xs.append(x)
            pis.append(pi)
            zis.append(zi)
        return np.array(xs), pis, zis

    def random_list3d(self, num):
        sample = {
            'xy': [],
            'zy': [],
            'xym': [],
            'zym': [],
            'xy_c': [],
            'zy_c': [],
            'xym_c': [],
            'zym_c': [],
            'xy_w': [],
            'zy_w': [],
            'xym_w': [],
            'zym_w': [],
            'p': [],
            'z': [],
        }
        
        for _ in range(num):
            p = []
            for y_max in self.y_num:
                p.append(self.rng.randint(y_max))
            sample['p'].append(p)
            z = [(pi/float(self.y_num[i]-1))*2-1 for i, pi in enumerate(p)] # [-1,1]
            sample['z'].append(z)

            file_path = self.list_from_p([p])[0]
            x, y = preprocess(file_path, self.data_type, self.x_range, self.y_range)

            xy = plane_view_np(x, xy_plane=True, project=True)
            zy = plane_view_np(x, xy_plane=False, project=True)
            xym = plane_view_np(x, xy_plane=True, project=False)
            zym = plane_view_np(x, xy_plane=False, project=False)

            sample['xy'].append(xy)
            sample['zy'].append(zy)
            sample['xym'].append(xym)
            sample['zym'].append(zym)

            if self.data_type == 'velocity':
                # vorticity
                x_c = np.expand_dims(x, axis=0)
                _, x_c = jacobian_np(x_c)
                x_c = self.to_vel(x_c)

                x_c = np.squeeze(x_c, axis=0)
                xy_c = plane_view_np(x_c, xy_plane=True, project=True)
                zy_c = plane_view_np(x_c, xy_plane=False, project=True)
                xym_c = plane_view_np(x_c, xy_plane=True, project=False)
                zym_c = plane_view_np(x_c, xy_plane=False, project=False)

                sample['xy_c'].append(xy_c)
                sample['zy_c'].append(zy_c)
                sample['xym_c'].append(xym_c)
                sample['zym_c'].append(zym_c)

            elif self.data_type == 'stream':
                # velocity
                x_c = np.expand_dims(x, axis=0)
                _, x_c = jacobian_np(x_c)
                x_c = self.to_vel(x_c)

                # vorticity
                _, x_w = jacobian_np(x_c)

                x_c = np.squeeze(x_c, axis=0)
                xy_c = plane_view_np(x_c, xy_plane=True, project=True)
                zy_c = plane_view_np(x_c, xy_plane=False, project=True)
                xym_c = plane_view_np(x_c, xy_plane=True, project=False)
                zym_c = plane_view_np(x_c, xy_plane=False, project=False)

                sample['xy_c'].append(xy_c)
                sample['zy_c'].append(zy_c)
                sample['xym_c'].append(xym_c)
                sample['zym_c'].append(zym_c)

                x_w = np.squeeze(x_w, axis=0)
                xy_w = plane_view_np(x_w, xy_plane=True, project=True)
                zy_w = plane_view_np(x_w, xy_plane=False, project=True)
                xym_w = plane_view_np(x_w, xy_plane=True, project=False)
                zym_w = plane_view_np(x_w, xy_plane=False, project=False)

                sample['xy_w'].append(xy_w)
                sample['zy_w'].append(zy_w)
                sample['xym_w'].append(xym_w)
                sample['zym_w'].append(zym_w)

        sample['xy'] = np.array(sample['xy'])
        sample['zy'] = np.array(sample['zy'])
        sample['xym'] = np.array(sample['xym'])
        sample['zym'] = np.array(sample['zym'])

        if self.data_type == 'velocity':
            sample['xy_c'] = np.array(sample['xy_c'])
            sample['zy_c'] = np.array(sample['zy_c'])
            sample['xym_c'] = np.array(sample['xym_c'])
            sample['zym_c'] = np.array(sample['zym_c'])

        elif self.data_type == 'stream':
            sample['xy_c'] = np.array(sample['xy_c'])
            sample['zy_c'] = np.array(sample['zy_c'])
            sample['xym_c'] = np.array(sample['xym_c'])
            sample['zym_c'] = np.array(sample['zym_c'])
            sample['xy_w'] = np.array(sample['xy_w'])
            sample['zy_w'] = np.array(sample['zy_w'])
            sample['xym_w'] = np.array(sample['xym_w'])
            sample['zym_w'] = np.array(sample['zym_w'])

        return sample

    def random_list(self, num):
        if self.is_3d:
            return self.random_list3d(num)
        else:
            return self.random_list2d(num)
    

def preprocess(file_path, data_type, x_range, y_range):    
    with np.load(file_path) as data:
        x = data['x']
        y = data['y']

    # horizontal flip
    if x.ndim == 4:
        # 3d
        # yxzd -> zyxd
        if data_type[0] == 'v' or data_type[0] == 's':
            x = x.transpose([2,0,1,3])
        x = x[:,::-1,:,:]
    else:
        x = x[::-1,:,:]

    # normalize
    if data_type[0] == 'd':
        x = x*2 - 1
    else:
        x /= x_range
        
    for i, ri in enumerate(y_range):
        y[i] = (y[i]-ri[0]) / (ri[1]-ri[0]) * 2 - 1
    return x, y

def test2d(config):
    prepare_dirs_and_logger(config)
    tf.set_random_seed(config.random_seed)

    batch_manager = BatchManager(config)

    # test: visualization
    # x, _ = preprocess('data/smoke_pos10_f100/v/4_50.npz', 'velocity', batch_manager.x_range, batch_manager.y_range)
    # x, _ = batch_manager.denorm(x=x)
    # x = x[::-1,:]

    # dudy = x[1:,:-1,0] - x[:-1,:-1,0]
    # dvdy = x[1:,:-1,1] - x[:-1,:-1,1]
    # dudx = x[:-1,1:,0] - x[:-1,:-1,0]
    # dvdx = x[:-1,1:,1] - x[:-1,:-1,1]
    
    # x_div = dudx + dvdy
    # x_vor = dvdx - dudy
    # print(x.min(), x.max(), x_div.min(), x_div.max(), x_vor.min(), x_vor.max())    
    
    # fig, ax = plt.subplots(2,2)
    # ax[0,0].imshow(x[:,:,0], cmap='viridis', origin='lower')
    # ax[0,1].imshow(x[:,:,1], cmap='viridis', origin='lower')
    # ax[1,0].imshow(x_div, cmap='viridis', origin='lower')
    # # ax[1,1].streamplot(X, Y, u, v, color=color, cmap=plt.cm.inferno, linewidth=lw,
    # #     density=2.0, arrowstyle='->', arrowsize=1.0)
    # # ax[1,1].set_aspect('equal')
    # ax[1,1].imshow(x_vor, cmap='RdBu', origin='lower')
    # plt.show()

    # thread test
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    sess_config.log_device_placement = False
    sess = tf.Session(config=sess_config)
    batch_manager.start_thread(sess)

    x, y = batch_manager.batch() # [-1, 1]
    x_ = x.eval(session=sess)
    # y_ = y.eval(session=sess)
    batch_manager.stop_thread()

    x_ = (x_+1)*127.5 # [0, 255]
    if config.data_type == 'velocity':
        b_ch = np.ones([config.batch_size,config.res_y,config.res_x,1])*127.5
        x_ = np.concatenate((x_, b_ch), axis=-1)

    save_image(x_, '{}/x_fixed.png'.format(config.model_dir))

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(x_[0,:,:,0], cmap='viridis', vmin=0, vmax=255) # viridis, gray
    # plt.subplot(122)
    # plt.imshow(x_[0,:,:,1], cmap='viridis', vmin=0, vmax=255)
    # plt.show()


    # random pick from parameter space
    x, pi, zi = batch_manager.random_list(8)
    save_image(x, '{}/x.png'.format(config.model_dir))
    with open('{}/x_p.txt'.format(config.model_dir), 'w') as f:
        f.write(str(pi))
        f.write(str(zi))

def test3d(config):
    prepare_dirs_and_logger(config)
    tf.set_random_seed(config.random_seed)

    batch_manager = BatchManager(config)
    x, _ = preprocess('data/smoke3_pos10_f100/v/4_50.npz', 'velocity', batch_manager.x_range, batch_manager.y_range)

    # batch test
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    sess_config.log_device_placement = False
    sess = tf.Session(config=sess_config)

    batch_manager.start_thread(sess)
    x, y = batch_manager.batch()
    x_ = x.eval(session=sess)    
    batch_manager.stop_thread()    

    # random pick from parameter space
    sample = batch_manager.random_list(8)    
    save_image(sample['xy'], '{}/xy.png'.format(config.model_dir))
    save_image(sample['zy'], '{}/zy.png'.format(config.model_dir))
    save_image(sample['xym'], '{}/xym.png'.format(config.model_dir))
    save_image(sample['zym'], '{}/zym.png'.format(config.model_dir))
    with open('{}/sample.txt'.format(config.model_dir), 'w') as f:
        f.write(str(sample['p']))
        f.write(str(sample['z']))

    if batch_manager.data_type == 'stream':
        save_image(sample['xy_c'], '{}/xy_c.png'.format(config.model_dir))
        save_image(sample['zy_c'], '{}/zy_c.png'.format(config.model_dir))
        save_image(sample['xym_c'], '{}/xym_c.png'.format(config.model_dir))
        save_image(sample['zym_c'], '{}/zym_c.png'.format(config.model_dir))        
        save_image(sample['xy_w'], '{}/xy_w.png'.format(config.model_dir))
        save_image(sample['zy_w'], '{}/zy_w.png'.format(config.model_dir))
        save_image(sample['xym_w'], '{}/xym_w.png'.format(config.model_dir))
        save_image(sample['zym_w'], '{}/zym_w.png'.format(config.model_dir))


if __name__ == "__main__":
    from config import get_config
    from utils import prepare_dirs_and_logger, save_config, save_image

    # # test: 2d
    # config, unparsed = get_config()
    # setattr(config, 'is_3d', False)
    # setattr(config, 'dataset', 'smoke_pos10_f100')
    # setattr(config, 'res_x', 96)
    # setattr(config, 'res_y', 128)
    # data_types = ['velocity']

    # for data_type in data_types:
    #     setattr(config, 'data_type', data_type)
    #     test2d(config)

    # test: 3d
    config, unparsed = get_config()
    setattr(config, 'is_3d', True)
    setattr(config, 'dataset', 'smoke3_pos10_f100')
    setattr(config, 'res_x', 48)
    setattr(config, 'res_y', 64)
    setattr(config, 'res_z', 48)
    setattr(config, 'batch_size', 4)
    data_types = ['velocity']

    for data_type in data_types:
        setattr(config, 'data_type', data_type)
        test3d(config)