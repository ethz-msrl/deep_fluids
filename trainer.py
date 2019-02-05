from __future__ import print_function

import os
import numpy as np
from tqdm import trange
from datetime import datetime
import time

from model import *
from util import *

class Trainer(object):
    def __init__(self, config, batch_manager):
        self.config = config

        self.batch_manager = batch_manager
        self.x, self.y = batch_manager.batch() # normalized input

        self.is_3d = config.is_3d
        self.dataset = config.dataset
        self.data_type = config.data_type
        if self.is_3d:
            self.x_jaco, self.x_vort = jacobian3(self.x)
        else:
            self.x_jaco, self.x_vort = jacobian(self.x)

        self.arch = config.arch
        self.res_x = config.res_x
        self.res_y = config.res_y
        self.res_z = config.res_z
        self.c_num = batch_manager.c_num
        self.b_num = config.batch_size
        self.test_b_num = config.test_batch_size

        self.repeat = config.repeat
        self.filters = config.filters
        self.num_conv = config.num_conv
        self.w1 = config.w1
        self.w2 = config.w2
        if 'dg' in self.arch: self.w3 = config.w3

        self.use_c = config.use_curl
        if self.use_c:
            if self.is_3d:
                self.output_shape = get_conv_shape(self.x)[1:-1] + [3]
            else:
                self.output_shape = get_conv_shape(self.x)[1:-1] + [1]
        else:
            self.output_shape = get_conv_shape(self.x)[1:]

        self.optimizer = config.optimizer
        self.beta1 = config.beta1
        self.beta2 = config.beta2
                
        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.start_step = config.start_step
        self.step = tf.Variable(self.start_step, name='step', trainable=False)
        # self.max_step = config.max_step
        self.max_step = int(config.max_epoch // batch_manager.epochs_per_step)

        self.lr_update = config.lr_update
        if self.lr_update == 'decay':
            lr_min = config.lr_min
            lr_max = config.lr_max
            self.g_lr = tf.Variable(lr_max, name='g_lr')
            self.g_lr_update = tf.assign(self.g_lr, 
               lr_min+0.5*(lr_max-lr_min)*(tf.cos(tf.cast(self.step, tf.float32)*np.pi/self.max_step)+1), name='g_lr_update')
        elif self.lr_update == 'step':
            self.g_lr = tf.Variable(config.lr_max, name='g_lr')
            self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr*0.5, config.lr_min), name='g_lr_update')    
        else:
            raise Exception("[!] Invalid lr update method")

        self.lr_update_step = config.lr_update_step
        self.log_step = config.log_step
        self.test_step = config.test_step
        self.save_sec = config.save_sec

        self.is_train = config.is_train
        self.build_model()

        self.saver = tf.train.Saver(max_to_keep=1000)
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=self.save_sec,
                                global_step=self.step,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)
        if self.is_train:
            self.batch_manager.start_thread(self.sess)

        # dirty way to bypass graph finilization error
        g = tf.get_default_graph()
        g._finalized = False

    def build_model(self):
        if self.use_c:
            self.G_s, self.G_var = GeneratorBE(self.y, self.filters, self.output_shape, 
                                               num_conv=self.num_conv, repeat=self.repeat)
            self.G_ = curl(self.G_s)
        else:
            self.G_, self.G_var = GeneratorBE(self.y, self.filters, self.output_shape,
                                              num_conv=self.num_conv, repeat=self.repeat)
        self.G = denorm_img(self.G_) # for debug

        self.G_jaco_, self.G_vort_ = jacobian(self.G_)
        self.G_vort = denorm_img(self.G_vort_)
        
        # to test
        self.z = tf.random_uniform(shape=[self.b_num, self.c_num], minval=-1.0, maxval=1.0)
        if self.use_c:
            self.G_z_s, _ = GeneratorBE(self.z, self.filters, self.output_shape,
                                        num_conv=self.num_conv, repeat=self.repeat, reuse=True)
            self.G_z_ = curl(self.G_z_s)
        else:
            self.G_z_, _ = GeneratorBE(self.z, self.filters, self.output_shape,
                                       num_conv=self.num_conv, repeat=self.repeat, reuse=True)
        self.G_z = denorm_img(self.G_z_) # for debug

        self.G_z_jaco_, self.G_z_vort_ = jacobian(self.G_z_)
        self.G_z_vort = denorm_img(self.G_z_vort_)

        if 'dg' in self.arch:
            # discriminator
            self.D_x, self.D_var = DiscriminatorPatch(self.x, self.filters)
            # D_in = tf.concat([self.x, self.x_jaco], axis=-1)
            # self.D_x, self.D_var = DiscriminatorPatch(D_in, self.filters)
            self.D_G, _ = DiscriminatorPatch(self.G_, self.filters, reuse=True)
            # G_in = tf.concat([self.G_, self.G_jaco_], axis=-1)
            # self.D_G, _ = DiscriminatorPatch(G_in, self.filters, reuse=True)
        
        show_all_variables()

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
            g_optimizer = optimizer(self.g_lr, beta1=self.beta1, beta2=self.beta2)
        elif self.optimizer == 'gd':
            optimizer = tf.train.GradientDescentOptimizer
            g_optimizer = optimizer(self.g_lr)
        else:
            raise Exception("[!] Invalid opimizer")

        # losses
        self.g_loss_l1 = tf.reduce_mean(tf.abs(self.G_ - self.x))
        self.g_loss_j_l1 = tf.reduce_mean(tf.abs(self.G_jaco_ - self.x_jaco))
        self.g_loss = self.g_loss_l1*self.w1 + self.g_loss_j_l1*self.w2
        
        if 'dg' in self.arch:
            self.g_loss_real = tf.reduce_mean(tf.square(self.D_G-1))
            self.d_loss_fake = tf.reduce_mean(tf.square(self.D_G))
            self.d_loss_real = tf.reduce_mean(tf.square(self.D_x-1))

            self.g_loss += self.g_loss_real*self.w3

            self.d_loss = self.d_loss_real + self.d_loss_fake
            self.d_optim = g_optimizer.minimize(self.d_loss, var_list=self.D_var)

        self.g_optim = g_optimizer.minimize(self.g_loss, global_step=self.step, var_list=self.G_var)
        self.epoch = tf.placeholder(tf.float32)

        # summary
        summary = [
            tf.summary.image("G", self.G),
            tf.summary.image("G_z", self.G_z),
            tf.summary.image("G_vort", self.G_vort),
            
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("loss/g_loss_l1", self.g_loss_l1),
            tf.summary.scalar("loss/g_loss_j_l1", self.g_loss_j_l1),

            tf.summary.scalar("misc/epoch", self.epoch),
            tf.summary.scalar('misc/q', self.batch_manager.q.size()),

            tf.summary.histogram("y", self.y),
            tf.summary.histogram("z", self.z),

            tf.summary.scalar("misc/g_lr", self.g_lr),
        ]

        if self.use_c:
            summary += [
                tf.summary.image("G_s", self.G_s[:,::-1]),
            ]

        if 'dg' in self.arch:
            summary += [
                tf.summary.scalar("loss/g_loss_real", tf.sqrt(self.g_loss_real)),
                tf.summary.scalar("loss/d_loss_real", tf.sqrt(self.d_loss_real)),
                tf.summary.scalar("loss/d_loss_fake", tf.sqrt(self.d_loss_fake)),
            ]

        self.summary_op = tf.summary.merge(summary)
        
        summary = [
            tf.summary.image("x", denorm_img(self.x)),
            tf.summary.image("x_vort", denorm_img(self.x_vort)),
        ]
        self.summary_once = tf.summary.merge(summary) # call just once

    def train(self):
        # test1: varying on each axis
        z_range = [-1, 1]
        z_shape = (self.b_num, self.c_num)
        z_samples = []
        z_varying = np.linspace(z_range[0], z_range[1], num=self.b_num)

        for i in range(self.c_num):
            zi = np.zeros(shape=z_shape)
            zi[:,i] = z_varying
            z_samples.append(zi)

        # test2: compare to gt
        x, pi, zi_ = self.batch_manager.random_list(self.b_num)
        x_w = self.get_vort_image(x/127.5-1)
        x_w = np.concatenate((x_w,x_w,x_w), axis=3)
        x = np.concatenate((x,x_w), axis=0)
        save_image(x, '{}/x_fixed_gt.png'.format(self.model_dir), nrow=self.b_num)

        with open('{}/x_fixed_gt.txt'.format(self.model_dir), 'w') as f:
            f.write(str(pi) + '\n')
            f.write(str(zi_))
        
        zi = np.zeros(shape=z_shape)            
        for i, z_gt in enumerate(zi_):
            zi[i,:] = z_gt
        z_samples.append(zi)

        # call once
        summary_once = self.sess.run(self.summary_once)
        self.summary_writer.add_summary(summary_once, 0)
        self.summary_writer.flush()
        
        # train
        for step in trange(self.start_step, self.max_step):
            if 'dg' in self.arch:
                self.sess.run([self.g_optim, self.d_optim])
            else:
                self.sess.run(self.g_optim)

            if step % self.log_step == 0 or step == self.max_step-1:
                ep = step*self.batch_manager.epochs_per_step
                loss, summary = self.sess.run([self.g_loss,self.summary_op],
                                              feed_dict={self.epoch: ep})
                assert not np.isnan(loss), 'Model diverged with loss = NaN'
                print("\n[{}/{}/ep{:.2f}] Loss: {:.6f}".format(step, self.max_step, ep, loss))

                self.summary_writer.add_summary(summary, global_step=step)
                self.summary_writer.flush()

            if step % self.test_step == 0 or step == self.max_step-1:
                self.generate(z_samples, self.model_dir, idx=step)

            if self.lr_update == 'step':
                if step % self.lr_update_step == self.lr_update_step - 1:
                    self.sess.run(self.g_lr_update)
            else:
                self.sess.run(self.g_lr_update)

        # save last checkpoint..
        save_path = os.path.join(self.model_dir, 'model.ckpt')
        self.saver.save(self.sess, save_path, global_step=self.step)
        self.batch_manager.stop_thread()

    def build_test_model(self):
        # build a model for testing
        self.z = tf.placeholder(dtype=tf.float32, shape=[self.test_b_num, self.c_num])
        if self.use_c:
            self.G_s, _ = GeneratorBE(self.z, self.filters, self.output_shape,
                                      num_conv=self.num_conv, repeat=self.repeat, reuse=True)
            self.G_ = curl(self.G_s)
        else:
            self.G_, _ = GeneratorBE(self.z, self.filters, self.output_shape,
                                     num_conv=self.num_conv, repeat=self.repeat, reuse=True)

    def test(self):
        self.build_test_model()
        
        p1, p2 = 10, 2

        # eval
        y1 = int(self.batch_manager.y_num[0])
        y2 = int(self.batch_manager.y_num[1])
        y3 = int(self.batch_manager.y_num[2])

        assert(y3 % self.test_b_num == 0)
        niter = int(y3 / self.test_b_num)

        c1 = p1/float(y1-1)*2-1
        c2 = p2/float(y2-1)*2-1

        z_range = [-1, 1]
        z_varying = np.linspace(z_range[0], z_range[1], num=y3)
        z_shape = (y3, self.c_num)

        z_c = np.zeros(shape=z_shape)
        z_c[:,0] = c1
        z_c[:,1] = c2
        z_c[:,-1] = z_varying

        G = []
        for b in range(niter):
            G_ = self.sess.run(self.G_, {self.z: z_c[self.test_b_num*b:self.test_b_num*(b+1),:]})
            G_, _ = self.batch_manager.denorm(x=G_)
            G.append(G_)
        G = np.concatenate(G, axis=0)

        # save
        title = '%d_%d' % (p1,p2)
        out_dir = os.path.join(self.model_dir, title)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for i, G_ in enumerate(G):
            dump_path = os.path.join(out_dir, '%d.npz' % i)
            np.savez_compressed(dump_path, x=G_)

    
    def generate(self, inputs, root_path=None, idx=None):
        generated = []
        for i, z_sample in enumerate(inputs):
            generated.append(self.sess.run(self.G, {self.y: z_sample}))
            
        c_concat = np.concatenate(tuple(generated[:-1]), axis=0)
        c_path = os.path.join(root_path, '{}_c.png'.format(idx))
        save_image(c_concat, c_path, nrow=self.b_num, flip=False)
        print("[*] Samples saved: {}".format(c_path))

        c_vort = self.get_vort_image(c_concat/127.5-1)
        c_path = os.path.join(root_path, '{}_cv.png'.format(idx))
        save_image(c_vort, c_path, nrow=self.b_num, flip=False)
        print("[*] Samples saved: {}".format(c_path))

        x = generated[-1]
        x_path = os.path.join(root_path, 'x_fixed_{}.png'.format(idx))
        x_w = self.get_vort_image(x/127.5-1)
        x_w = np.concatenate((x_w,x_w,x_w), axis=3)
        x = np.concatenate((x,x_w), axis=0)

        save_image(x, x_path, nrow=self.b_num, flip=False)
        print("[*] Samples saved: {}".format(x_path))

    def get_vort_image(self, x):
        x = vort_np(x[:,:,:,:2])
        x_img = np.clip((x+1)*127.5, 0, 255)
        return x_img