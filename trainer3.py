from __future__ import print_function

import os
import numpy as np
from tqdm import trange
from datetime import datetime
import time

from models import *
from utils import save_image, convert_png2mp4, streamplot, vortplot, gradplot, jacoplot, divplot
from trainer import Trainer

class Trainer3(Trainer):
    def build_model(self):
        if self.use_c:
            self.G_s, self.G_var = GeneratorBE3(self.y, self.filters, self.output_shape, 
                                               num_conv=self.num_conv, last_k=self.last_k,
                                               repeat=self.repeat, skip_concat=self.skip_concat,
                                               act=self.act)
            _, self.G_ = jacobian3(self.G_s)
        else:
            self.G_, self.G_var = GeneratorBE3(self.y, self.filters, self.output_shape,
                                              num_conv=self.num_conv, last_k=self.last_k, 
                                              repeat=self.repeat, skip_concat=self.skip_concat,
                                              act=self.act)
        self.G = denorm_img3(self.G_) # for debug

        self.G_jaco_, self.G_vort_ = jacobian3(self.G_)
        self.G_vort = denorm_img3(self.G_vort_)
        self.G_div_ = divergence3(self.G_*self.batch_manager.x_range)
        
        # to test
        self.z = tf.random_uniform(shape=[self.b_num, self.c_num], minval=-1.0, maxval=1.0)
        if self.use_c:
            self.G_z_s, _ = GeneratorBE3(self.z, self.filters, self.output_shape,
                                        num_conv=self.num_conv, last_k=self.last_k,
                                        repeat=self.repeat, skip_concat=self.skip_concat,
                                        act=self.act, reuse=True)
            _, self.G_z_ = jacobian3(self.G_z_s)
        else:
            self.G_z_, _ = GeneratorBE3(self.z, self.filters, self.output_shape,
                                       num_conv=self.num_conv, last_k=self.last_k,
                                       repeat=self.repeat, skip_concat=self.skip_concat,
                                       act=self.act, reuse=True)
        self.G_z = denorm_img3(self.G_z_) # for debug

        self.G_z_jaco_, self.G_z_vort_ = jacobian3(self.G_z_)
        self.G_z_vort = denorm_img3(self.G_z_vort_)
        self.G_z_div_ = divergence3(self.G_z_*self.batch_manager.x_range)
        
        show_all_variables()

        if self.lr_update == 'freeze':
            num_layers = int(self.G_var[-1].name.split('_')[0].split('/')[1])+1
            layers = []
            t0 = 0.5 # 0.5 for linear
            is_cubic = True
            t_ = np.linspace(t0, 1, num_layers, dtype=np.float32)
            if is_cubic: t_ = np.power(t_, 3)
            self.max_steps = (self.max_step * t_).astype(np.int)
            a_ = self.lr_max / t_
            self.g_lr_update = []
            for i in range(num_layers):
                g_vars = [var for var in self.G_var if '/{}_'.format(i) in var.name]
                lr = tf.Variable(a_[i], name='lr%02d' % i)
                layers.append({
                    'v': g_vars, 
                    'lr': lr,
                })
                lr_update = tf.assign(lr,
                    0.5*a_[i]*(tf.cos(tf.cast(self.step, tf.float32)*np.pi/self.max_steps[i])+1))
                self.g_lr_update.append(lr_update)
            self.max_steps = self.max_steps.tolist()
        else:
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

        self.g_loss_ke = tf.abs(tf.reduce_mean(tf.square(self.G_)) - tf.reduce_mean(tf.square(self.x)))
        self.g_loss_div = tf.reduce_mean(tf.abs(self.G_div_))
        self.g_loss_div_max = tf.reduce_max(tf.abs(self.G_div_))
        self.g_loss_z_div = tf.reduce_mean(tf.abs(self.G_z_div_))
        self.g_loss_z_div_max = tf.reduce_max(tf.abs(self.G_z_div_))

        self.g_loss = self.g_loss_l1*self.w1 + self.g_loss_j_l1*self.w2

        if self.lr_update == 'freeze':
            self.opts = []
            for i in range(num_layers):
                grad = tf.gradients(self.g_loss, layers[i]['v'])
                opt = tf.train.AdamOptimizer(layers[i]['lr'], beta1=self.beta1, beta2=self.beta2)
                if i == num_layers-1:
                    self.opts.append(opt.apply_gradients(zip(grad, layers[i]['v']), global_step=self.step))
                else:
                    self.opts.append(opt.apply_gradients(zip(grad, layers[i]['v'])))
            self.g_optim = tf.group(self.opts)
        else:
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

            tf.summary.scalar("loss/g_loss_ke", self.g_loss_ke),           
            tf.summary.scalar("loss/g_loss_div", self.g_loss_div),
            tf.summary.scalar("loss/g_loss_div_max", self.g_loss_div_max),
            tf.summary.scalar("loss/g_loss_z_div", self.g_loss_z_div),
            tf.summary.scalar("loss/g_loss_z_div_max", self.g_loss_z_div_max),

            tf.summary.scalar("misc/epoch", self.epoch),
            tf.summary.scalar('misc/q', self.batch_manager.q.size()),

            tf.summary.histogram("y", self.y),
            tf.summary.histogram("z", self.z),
        ]

        # if self.use_c:
        #     summary += [
        #         tf.summary.image("G_s", self.G_s),
        #     ]

        if self.lr_update == 'freeze':
            for i in range(num_layers):
                summary.append(tf.summary.scalar("g_lr/%02d" % i, layers[i]['lr']))
        else:
            summary += [
                tf.summary.scalar("misc/g_lr", self.g_lr),
            ]

        self.summary_op = tf.summary.merge(summary)

        # summary once
        x = denorm_img3(self.x)
        x_vort = denorm_img3(self.x_vort)
        
        x_div = divergence3(self.x*self.batch_manager.x_range)
        x_div_mean = tf.reduce_mean(tf.abs(x_div))
        x_div_max = tf.reduce_max(tf.abs(x_div))
        
        x_shape = int_shape(x_div) # bzyxd
        c_id = [int(x_shape[1]/2), int(x_shape[3]/2)]
        x_div_m = tf.squeeze(tf.slice(x_div, [0,c_id[0],0,0,0], [-1,1,-1,-1,-1]), [1])  

        summary = [
            # tf.summary.image("x/xy", x['xy']),
            # tf.summary.image("x/zy", x['zy']),
            tf.summary.image("x/xym", x['xym']),
            tf.summary.image("x/zym", x['zym']),

            # tf.summary.image("x_vort/xy", x_vort['xy']),
            # tf.summary.image("x_vort/zy", x_vort['zy']),
            tf.summary.image("x_vort/xym", x_vort['xym']),
            tf.summary.image("x_vort/zym", x_vort['zym']),
            
            tf.summary.scalar('x_div/mean', x_div_mean),
            tf.summary.scalar('x_div/max', x_div_max),
          
            tf.summary.image("x_div/m", x_div_m),
        ]
        self.summary_once = tf.summary.merge(summary) # call just once

    def train(self):
        # test1: varying on each axis
        z_range = [-1, 1]
        z_shape = (self.test_batch_size, self.c_num)
        z_samples = []
        z_varying = np.linspace(z_range[0], z_range[1], num=self.test_batch_size)

        for i in range(self.c_num):
            zi = np.zeros(shape=z_shape)
            # if i == self.c_num-1:
            #     zi = np.zeros(shape=z_shape)
            # else:
            #     zi = np.ones(shape=z_shape)*-1
            zi[:,i] = z_varying
            z_samples.append(zi)

        # test2: compare to gt
        gen_list = self.batch_manager.random_list(self.test_batch_size)
        x_xy = np.concatenate((gen_list['xym'],gen_list['xym_c']), axis=0)
        x_zy = np.concatenate((gen_list['zym'],gen_list['zym_c']), axis=0)
        save_image(x_xy, '{}/x_fixed_xym_gt.png'.format(self.model_dir), padding=1, nrow=self.test_batch_size)
        save_image(x_zy, '{}/x_fixed_zym_gt.png'.format(self.model_dir), padding=1, nrow=self.test_batch_size)
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
            if self.lr_update == 'freeze' and step == self.max_steps[0]:
                self.max_steps.pop(0)
                self.opts.pop(0)
                self.g_optim = tf.group(self.opts)
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
                g_lr = self.sess.run(self.g_lr_update)
                # print(g_lr)

        # save last checkpoint..
        save_path = os.path.join(self.model_dir, 'model.ckpt')
        self.saver.save(self.sess, save_path, global_step=self.step)
        self.batch_manager.stop_thread()
    
    def build_test_model(self, b_num):
        self.zt = tf.placeholder(dtype=tf.float32, shape=[b_num, self.c_num])
        if self.use_c:
            self.Gt_s, _ = GeneratorBE3(self.zt, self.filters, self.output_shape,
                                       num_conv=self.num_conv, last_k=self.last_k,
                                       repeat=self.repeat, skip_concat=self.skip_concat,
                                       act=self.act, reuse=True)
            _, self.Gt_ = jacobian3(self.Gt_s)
        else:
            self.Gt_, _ = GeneratorBE3(self.zt, self.filters, self.output_shape,
                                       num_conv=self.num_conv, last_k=self.last_k,
                                       repeat=self.repeat, skip_concat=self.skip_concat,
                                       act=self.act, reuse=True)
        self.Gt = denorm_img3(self.Gt_) # for debug

        _, self.Gt_vort_ = jacobian3(self.Gt_)
        self.Gt_vort = denorm_img3(self.Gt_vort_)

    def test(self):
        self.build_test_model(self.test_batch_size)
        
        self.b_num = self.test_batch_size
        self.z = self.zt
        self.G_ = self.Gt_
        self.G = self.Gt

        # self.test_smokegun()
        self.test_smokeobs()
        return

    def test_smokegun(self):
        p_list = [
            [0,0], [0,1], [0,2],
            [1,0], [1,1], [1,2],
            [2,0], [2,1], [2,2],
            [3,0], [3,1], [3,2],
            [4,0], [4,1], [4,2],
            [0,0.5], [0,1.5], [3.5,2],
        ]

        for p12 in p_list:
            print(p12)            
            p1_, p2_ = p12[0], p12[1]
            out_dir = os.path.join(self.model_dir, 'p2_n%d' % self.test_intv)
            title = str(p1_) + '_' + str(p2_)
            dump_path = os.path.join(out_dir, title+'.npz')
            
            G = self.gen_p2(p1_, p2_)
            G = G[:,:,::-1,:,:]
            G = G.transpose([0,2,3,1,4]) # bzyxd -> byxzd
            np.savez_compressed(dump_path, v=G)

            from subprocess import call
            call(["../manta/build_nogui_vdb/Release/manta.exe",
                    "./scene/smoke3_vel_buo.py",
                    "--is_test=True",
                    "--vpath={}".format(dump_path)])

    def test_smokeobs(self):
        p_list = [
            [4.5,2],
            [0,2], [2,2], [5,2], [8,2], [10,2],
            [4,2], 
        ]

        for p12 in p_list:
            print(p12)            
            p1_, p2_ = p12[0], p12[1]
            out_dir = os.path.join(self.model_dir, 'p2_n%d' % self.test_intv)
            title = str(p1_) + '_' + str(p2_)
            dump_path = os.path.join(out_dir, title+'.npz')
            
            G = self.gen_p2(p1_, p2_)
            G = G[:,:,::-1,:,:]
            G = G.transpose([0,2,3,1,4]) # bzyxd -> byxzd
            np.savez_compressed(dump_path, v=G)

            from subprocess import call
            call(["../manta/build_nogui_vdb/Release/manta.exe",
                    "./scene/smoke3_obs_buo.py",
                    "--is_test=True",
                    "--vpath={}".format(dump_path)])

    def generate(self, inputs, root_path=None, idx=None):
        # xy_list = []
        # zy_list = []
        xym_list = []
        zym_list = []

        # xyw_list = []
        # zyw_list = []
        xymw_list = []
        zymw_list = []

        for i, z_sample in enumerate(inputs):
            xym, zym = self.sess.run( # xy, zy, 
                [self.Gt['xym'], self.Gt['zym']], {self.zt: z_sample}) # self.Gt['xy'], self.Gt['zy'], 
            # xy_list.append(xy)
            # zy_list.append(zy)
            xym_list.append(xym)
            zym_list.append(zym)

            xym, zym = self.sess.run( # xy, zy, 
                [self.Gt_vort['xym'], self.Gt_vort['zym']], {self.zt: z_sample}) # self.Gt_vort['xy'], self.Gt_vort['zy'], 
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
            save_image(c_concat, c_path, nrow=self.test_batch_size, padding=1)
            print("[*] Samples saved: {}".format(c_path))

        gen_random = np.concatenate(tuple(xym_list[-2:]), axis=0)
        x_xy_path = os.path.join(root_path, 'x_fixed_xym_{}.png'.format(idx))
        save_image(gen_random, x_xy_path, nrow=self.test_batch_size, padding=1)
        print("[*] Samples saved: {}".format(x_xy_path))

        gen_random = np.concatenate(tuple(zym_list[-2:]), axis=0)
        x_zy_path = os.path.join(root_path, 'x_fixed_zym_{}.png'.format(idx))
        save_image(gen_random, x_zy_path, nrow=self.test_batch_size, padding=1)
        print("[*] Samples saved: {}".format(x_zy_path))