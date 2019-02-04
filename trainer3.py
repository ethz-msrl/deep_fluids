from __future__ import print_function

import os
import numpy as np
from tqdm import trange
from datetime import datetime
import time

from model import *
from util import *
from trainer import Trainer

class Trainer3(Trainer):
    def build_model(self):
        if self.use_c:
            self.G_s, self.G_var = GeneratorBE3(self.y, self.filters, self.output_shape, 
                                               num_conv=self.num_conv, repeat=self.repeat)
            _, self.G_ = jacobian3(self.G_s)
        else:
            self.G_, self.G_var = GeneratorBE3(self.y, self.filters, self.output_shape,
                                              num_conv=self.num_conv, repeat=self.repeat)
        self.G = denorm_img3(self.G_) # for debug

        self.G_jaco_, self.G_vort_ = jacobian3(self.G_)
        self.G_vort = denorm_img3(self.G_vort_)
        
        # to test
        self.z = tf.random_uniform(shape=[self.b_num, self.c_num], minval=-1.0, maxval=1.0)
        if self.use_c:
            self.G_z_s, _ = GeneratorBE3(self.z, self.filters, self.output_shape,
                                        num_conv=self.num_conv, repeat=self.repeat, reuse=True)
            _, self.G_z_ = jacobian3(self.G_z_s)
        else:
            self.G_z_, _ = GeneratorBE3(self.z, self.filters, self.output_shape,
                                       num_conv=self.num_conv, repeat=self.repeat, reuse=True)
        self.G_z = denorm_img3(self.G_z_) # for debug

        self.G_z_jaco_, self.G_z_vort_ = jacobian3(self.G_z_)
        self.G_z_vort = denorm_img3(self.G_z_vort_)
        
        if 'dg' in self.arch:
            # discriminator
            self.D_x, self.D_var = DiscriminatorPatch3(self.x, self.filters)
            self.D_G, _ = DiscriminatorPatch3(self.G_, self.filters, reuse=True)

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
            # tf.summary.image("G/xy", self.G['xy']),
            # tf.summary.image("G/zy", self.G['zy']),
            tf.summary.image("G/xym", self.G['xym']),
            tf.summary.image("G/zym", self.G['zym']),
            
            # tf.summary.image("G_z/xy", self.G_z['xy']),
            # tf.summary.image("G_z/zy", self.G_z['zy']),
            tf.summary.image("G_z/xym", self.G_z['xym']),
            tf.summary.image("G_z/zym", self.G_z['zym']),
            
            # tf.summary.image("G_vort/xy", self.G_vort['xy']),
            # tf.summary.image("G_vort/zy", self.G_vort['zy']),
            tf.summary.image("G_vort/xym", self.G_vort['xym']),
            tf.summary.image("G_vort/zym", self.G_vort['zym']),
            
            # tf.summary.image("G_z_vort/xy", self.G_z_vort['xy']),
            # tf.summary.image("G_z_vort/zy", self.G_z_vort['zy']),
            tf.summary.image("G_z_vort/xym", self.G_z_vort['xym']),
            tf.summary.image("G_z_vort/zym", self.G_z_vort['zym']),

            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("loss/g_loss_l1", self.g_loss_l1),
            tf.summary.scalar("loss/g_loss_j_l1", self.g_loss_j_l1),

            tf.summary.scalar("misc/epoch", self.epoch),
            tf.summary.scalar('misc/q', self.batch_manager.q.size()),

            tf.summary.histogram("y", self.y),
            tf.summary.histogram("z", self.z),

            tf.summary.scalar("misc/g_lr", self.g_lr),
        ]

        if 'dg' in self.arch:
            summary += [
                tf.summary.scalar("loss/g_loss_real", tf.sqrt(self.g_loss_real)),
                tf.summary.scalar("loss/d_loss_real", tf.sqrt(self.d_loss_real)),
                tf.summary.scalar("loss/d_loss_fake", tf.sqrt(self.d_loss_fake)),
            ]

        self.summary_op = tf.summary.merge(summary)

        # summary once
        x = denorm_img3(self.x)
        x_vort = denorm_img3(self.x_vort)
        
        summary = [
            # tf.summary.image("x/xy", x['xy']),
            # tf.summary.image("x/zy", x['zy']),
            tf.summary.image("x/xym", x['xym']),
            tf.summary.image("x/zym", x['zym']),

            # tf.summary.image("x_vort/xy", x_vort['xy']),
            # tf.summary.image("x_vort/zy", x_vort['zy']),
            tf.summary.image("x_vort/xym", x_vort['xym']),
            tf.summary.image("x_vort/zym", x_vort['zym']),
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
        gen_list = self.batch_manager.random_list(self.b_num)
        x_xy = np.concatenate((gen_list['xym'],gen_list['xym_c']), axis=0)
        x_zy = np.concatenate((gen_list['zym'],gen_list['zym_c']), axis=0)
        save_image(x_xy, '{}/x_fixed_xym_gt.png'.format(self.model_dir), padding=1, nrow=self.b_num)
        save_image(x_zy, '{}/x_fixed_zym_gt.png'.format(self.model_dir), padding=1, nrow=self.b_num)
        with open('{}/x_fixed_gt.txt'.format(self.model_dir), 'w') as f:
            f.write(str(gen_list['p']) + '\n')
            f.write(str(gen_list['z']))

        zi = np.zeros(shape=z_shape)            
        for i, z_gt in enumerate(gen_list['z']):
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
            self.G_s, _ = GeneratorBE3(self.z, self.filters, self.output_shape,
                                      num_conv=self.num_conv, repeat=self.repeat, reuse=True)
            self.G_ = curl(self.G_s)
        else:
            self.G_, _ = GeneratorBE3(self.z, self.filters, self.output_shape,
                                     num_conv=self.num_conv, repeat=self.repeat, reuse=True)        
    
    def generate(self, inputs, root_path=None, idx=None):
        # xy_list = []
        # zy_list = []
        xym_list = []
        zym_list = []

        # xyw_list = []
        # zyw_list = []
        xymw_list = []
        zymw_list = []

        for _, z_sample in enumerate(inputs):
            xym, zym = self.sess.run( # xy, zy, 
                [self.G['xym'], self.G['zym']], {self.z: z_sample}) # self.G['xy'], self.G['zy'], 
            # xy_list.append(xy)
            # zy_list.append(zy)
            xym_list.append(xym)
            zym_list.append(zym)

            xym, zym = self.sess.run( # xy, zy, 
                [self.G_vort['xym'], self.G_vort['zym']], {self.z: z_sample}) # self.G_vort['xy'], self.G_vort['zy'], 
            # xyw_list.append(xy)
            # zyw_list.append(zy)
            xymw_list.append(xym)
            zymw_list.append(zym)

        xym_list = xym_list[:-1] + xymw_list[:-1] + [xym_list[-1]] + [xymw_list[-1]]
        zym_list = zym_list[:-1] + zymw_list[:-1] + [zym_list[-1]] + [zymw_list[-1]]

        for tag, generated in zip(['xym','zym'], # '0_xy','0_zy',
                                  [xym_list, zym_list]): # xy_list, zy_list, 
            c_concat = np.concatenate(tuple(generated[:-2]), axis=0)
            c_path = os.path.join(root_path, '{}_{}.png'.format(idx,tag))
            save_image(c_concat, c_path, nrow=self.b_num, padding=1)
            print("[*] Samples saved: {}".format(c_path))

        gen_random = np.concatenate(tuple(xym_list[-2:]), axis=0)
        x_xy_path = os.path.join(root_path, 'x_fixed_xym_{}.png'.format(idx))
        save_image(gen_random, x_xy_path, nrow=self.b_num, padding=1)
        print("[*] Samples saved: {}".format(x_xy_path))

        gen_random = np.concatenate(tuple(zym_list[-2:]), axis=0)
        x_zy_path = os.path.join(root_path, 'x_fixed_zym_{}.png'.format(idx))
        save_image(gen_random, x_zy_path, nrow=self.b_num, padding=1)
        print("[*] Samples saved: {}".format(x_zy_path))