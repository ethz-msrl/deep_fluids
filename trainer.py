from __future__ import print_function

import os
import numpy as np
from tqdm import trange
from datetime import datetime
import time

from models import *
from utils import save_image, convert_png2mp4, streamplot, vortplot, gradplot, jacoplot

class Trainer(object):
    def __init__(self, config, batch_manager):
        self.config = config

        self.batch_manager = batch_manager
        self.x, self.y = batch_manager.batch() # normalized input

        self.dataset = config.dataset
        self.data_type = config.data_type
        if self.data_type == 'velocity':
            self.x_jaco, self.x_vort = jacobian(self.x)
        self.output_shape = get_conv_shape(self.x)[1:]

        self.is_3d = config.is_3d
        self.archi = config.archi
        self.res_x = config.res_x
        self.res_y = config.res_y
        self.res_z = config.res_z
        self.c_num = batch_manager.c_num
        self.b_num = config.batch_size

        self.filters = config.filters
        self.w1 = config.w1
        self.w2 = config.w2
        self.use_j = config.use_jaco
        self.w_adv = config.w_adv

        self.optimizer = config.optimizer
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.g_lr = tf.Variable(config.g_lr, name='g_lr')
        self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr*0.5, config.lr_lower_boundary), name='g_lr_update')        
        if self.archi == 'dg':
            self.d_lr = tf.Variable(config.d_lr, name='d_lr')
            self.d_lr_update = tf.assign(self.d_lr, tf.maximum(self.d_lr*0.5, config.lr_lower_boundary), name='d_lr_update')

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.start_step = config.start_step
        self.step = tf.Variable(self.start_step, name='step', trainable=False)
        self.max_step = config.max_step
        self.lr_update_step = config.lr_update_step
        self.log_step = config.log_step
        self.test_step = config.test_step
        self.save_sec = config.save_sec
        self.test_intv = config.test_intv
        self.test_batch_size = config.test_batch_size

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

    def build_model(self):
        self.G_, self.G_var = GeneratorBE(self.y, self.filters, self.output_shape)
        self.G = denorm_img(self.G_) # for debug

        if self.data_type == 'velocity':
            self.G_jaco_, self.G_vort_ = jacobian(self.G_)
            self.G_vort = denorm_img(self.G_vort_)
            self.G_div_ = divergence(self.G_*self.batch_manager.x_range)
                
        # to test
        self.z = tf.random_uniform(shape=[self.b_num, self.c_num], minval=-1.0, maxval=1.0)
        self.G_z_, _ = GeneratorBE(self.z, self.filters, self.output_shape, reuse=True)
        self.G_z = denorm_img(self.G_z_) # for debug

        if self.data_type == 'velocity':
            self.G_z_jaco_, self.G_z_vort_ = jacobian(self.G_z_)
            self.G_z_vort = denorm_img(self.G_z_vort_)
            self.G_z_div_ = divergence(self.G_z_*self.batch_manager.x_range)
        
        if self.archi == 'dg':
            x_r_disc = self.x
            x_f_disc = self.G_z_
            if self.use_j:
                x_r_disc = self.x_jaco
                x_f_disc = self.G_z_jaco_

            self.D_x, _, self.D_var = DiscriminatorPatch(x_r_disc, self.filters, 'D')
            self.D_G, _, _ = DiscriminatorPatch(x_f_disc, self.filters, 'D', reuse=True)
        
        show_all_variables()

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(self.config.optimizer))

        g_optimizer = optimizer(self.g_lr, beta1=self.beta1, beta2=self.beta2)
        
        # losses
        self.g_loss_l1 = tf.reduce_mean(tf.abs(self.G_ - self.x))
        if self.data_type == 'velocity':
            self.g_loss_j_l1 = tf.reduce_mean(tf.abs(self.G_jaco_ - self.x_jaco))
            
            self.g_loss_ke = tf.abs(tf.reduce_mean(tf.square(self.G_)) - tf.reduce_mean(tf.square(self.x)))
            self.g_loss_div = tf.reduce_mean(tf.abs(self.G_div_))
            self.g_loss_div_max = tf.reduce_max(tf.abs(self.G_div_))
            self.g_loss_z_div = tf.reduce_mean(tf.abs(self.G_z_div_))
            self.g_loss_z_div_max = tf.reduce_max(tf.abs(self.G_z_div_))

            self.g_loss = self.g_loss_l1*self.w1 + self.g_loss_j_l1*self.w2
        else:
            self.g_loss = self.g_loss_l1

        if self.archi == 'dg':
            d_optimizer = optimizer(self.d_lr, beta1=self.beta1, beta2=self.beta2)
            self.g_loss_real = tf.reduce_mean(tf.square(self.D_G-1))
            self.d_loss_fake = tf.reduce_mean(tf.square(self.D_G))
            self.d_loss_real = tf.reduce_mean(tf.square(self.D_x-1))

            self.g_loss += self.g_loss_real*self.w_adv
            self.d_loss = self.d_loss_fake + self.d_loss_real
            self.d_optim = d_optimizer.minimize(self.d_loss, var_list=self.D_var)

        self.g_optim = g_optimizer.minimize(self.g_loss, global_step=self.step, var_list=self.G_var)
        self.epoch = tf.placeholder(tf.float32)

        # summary
        summary = [
            tf.summary.image("G", self.G),
            tf.summary.image("G_z", self.G_z),

            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("loss/g_loss_l1", self.g_loss_l1),
           
            tf.summary.scalar("misc/g_lr", self.g_lr),
            tf.summary.scalar("misc/epoch", self.epoch),
            tf.summary.scalar('misc/q', self.batch_manager.q.size()),

            tf.summary.histogram("y", self.y),
            tf.summary.histogram("z", self.z),
        ]

        if self.archi == 'dg':
            summary += [
                tf.summary.scalar("loss/g_loss_real", self.g_loss_real),
                tf.summary.scalar("loss/d_loss", self.d_loss),
                tf.summary.scalar("loss/d_loss_real", self.d_loss_real),
                tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
                tf.summary.scalar("misc/d_lr", self.d_lr),
            ]

        if self.data_type == 'velocity':
            summary += [
                tf.summary.image("G_vort", self.G_vort),
                tf.summary.image("G_div", self.G_div_),
                tf.summary.image("G_z_div", self.G_z_div_),

                tf.summary.scalar("loss/g_loss_ke", self.g_loss_ke),
                tf.summary.scalar("loss/g_loss_j_l1", self.g_loss_j_l1),
                
                tf.summary.scalar("loss/g_loss_div", self.g_loss_div),
                tf.summary.scalar("loss/g_loss_div_max", self.g_loss_div_max),
                tf.summary.scalar("loss/g_loss_z_div", self.g_loss_z_div),
                tf.summary.scalar("loss/g_loss_z_div_max", self.g_loss_z_div_max),
            ]

        self.summary_op = tf.summary.merge(summary)

        summary = [
            tf.summary.image("x", denorm_img(self.x)),
        ]

        if self.data_type == 'velocity':
            summary += [
                tf.summary.image("x_vort", denorm_img(self.x_vort)),
                tf.summary.image('x_div', divergence(self.x*self.batch_manager.x_range)),
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
            # if i == self.c_num-1:
            #     zi = np.zeros(shape=z_shape)
            # else:
            #     zi = np.ones(shape=z_shape)*-1
            zi[:,i] = z_varying
            z_samples.append(zi)

        # test2: compare to gt
        x, pi, zi_ = self.batch_manager.random_list(self.b_num)
        if self.data_type == 'velocity':
            x_w = self.get_vort_image(x/127.5-1, is_vel=True)
            x_w = np.concatenate((x_w,x_w,x_w), axis=3)
            x = np.concatenate((x,x_w), axis=0)
        save_image(x, '{}/x_fixed_gt.png'.format(self.model_dir))

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
        optim = [self.g_optim]
        lr_update = [self.g_lr_update]
        if self.archi == 'dg':
            optim.append(self.d_optim)
            lr_update.append(self.d_lr_update)
        
        for step in trange(self.start_step, self.max_step):
            self.sess.run(optim)

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

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run(lr_update)

        # save last checkpoint..
        save_path = os.path.join(self.model_dir, 'model.ckpt')
        self.saver.save(self.sess, save_path, global_step=self.step)
        self.batch_manager.stop_thread()
    
    def build_test_model(self, b_num):
        self.b_num = b_num
        self.z = tf.placeholder(dtype=tf.float32, shape=[self.b_num, self.c_num])
        self.G_, _ = GeneratorBE(self.z, self.filters, self.output_shape, reuse=True)
        self.G = denorm_img(self.G_) # for debug

    def test(self):
        # dirty way to bypass graph finilization error
        g = tf.get_default_graph()
        g._finalized = False

        self.build_test_model(self.test_batch_size)

        intv = self.test_intv
        z_range = [-1, 1]
        z_varying = np.linspace(z_range[0], z_range[1], num=intv)
        z_shape = (intv, self.c_num)

        from itertools import product
        for p in range(self.c_num):
            out_dir = os.path.join(self.model_dir, 'p%d_n%d' % (p, intv))

            c_list = []
            p_list = []
            for i in range(self.c_num):
                if i != p:
                    y_num = int(self.batch_manager.y_num[i])
                    if y_num < 5:
                        p_ = range(y_num)
                    else:
                        p_ = [0, 1, int(y_num/2)-1, int(y_num/2), y_num-2, y_num-1]

                    p_list.append(p_)
                    c_list.append([y/float(y_num-1)*2-1 for y in p_])

            dump = (p == self.c_num-1)
            for c, ps in zip(product(*c_list), product(*p_list)):
                title = ('%d_'*len(ps) % ps)[:-1]
                c = list(c)
                c.insert(p, z_varying)

                z_c = np.zeros(shape=z_shape)
                for i in range(self.c_num):
                    z_c[:,i] = c[i]
                
                self.generate_video(title, out_dir, z_c, dump=dump)

            if p == self.c_num-1:
                # interpolation test
                # p_list = []
                p1 = 9.5
                p2 = 1
                y1 = int(self.batch_manager.y_num[0])
                y2 = int(self.batch_manager.y_num[1])

                c1 = p1/float(y1-1)*2-1
                c2 = p2/float(y2-1)*2-1

                z_c = np.zeros(shape=z_shape)
                z_c[:,0] = c1
                z_c[:,1] = c2
                z_c[:,-1] = z_varying
                title = str(p1) + '_' + str(p2)
                self.generate_video(title, out_dir, z_c, dump=True)
                
    def generate_video(self, title, out_dir, z_c, dump=False):
        img_dir = os.path.join(out_dir, title)
        print(img_dir)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        if self.data_type == 'stream':
            img_c_dir = os.path.join(out_dir, title+'c')
            if not os.path.exists(img_c_dir):
                os.makedirs(img_c_dir)

        intv = z_c.shape[0]
        assert(intv % self.b_num == 0)
        niter = int(intv / self.b_num)


        # self.sess.run(self.G_, {self.z: z_c[:self.b_num,:]})

        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        # self.sess.run(self.G_, {self.z: z_c[:self.b_num,:]}, options=run_options, run_metadata=run_metadata)

        # # Create the Timeline object, and write it to a json
        # from tensorflow.python.client import timeline
        # tl = timeline.Timeline(run_metadata.step_stats)
        # ctf = tl.generate_chrome_trace_format()
        # with open('timeline.json', 'w') as f:
        #     f.write(ctf)

        # assert(False)
        # return

        from datetime import datetime
        import time

        time_path = os.path.join(out_dir, title+'.txt')
        f = open(time_path, 'w')

        # for b in range(niter):
        #     start_time = time.time()
        #     G = self.sess.run(self.G, {self.z: z_c[self.b_num*b:self.b_num*(b+1),:]})
        #     duration = time.time() - start_time
        #     duration_str = '%s: %.3f sec/%d frames (%.3f ms/frame)' % (
        #         datetime.now(), duration, self.b_num, duration/self.b_num*1000.0)
        #     print(duration_str)
        #     f.write(duration_str+'\n')
        #     if self.data_type == 'stream':
        #         G_curl = self.sess.run(self.G_curl, {self.z: z_c[self.b_num*b:self.b_num*(b+1),:]})

        #     for i in range(self.b_num):
        #         x = G[i]
        #         img_path = os.path.join(img_dir, '%04d.png' % (self.b_num*b + i))
        #         save_image(x, img_path, single=True)

        #         if self.data_type == 'stream':
        #             x = G_curl[i]
        #             img_path = os.path.join(img_c_dir, '%04d.png' % (self.b_num*b + i))
        #             save_image(x, img_path, single=True)
        
        # f.close()
        # mp4_path = os.path.join(out_dir, title+'.mp4')
        # fps = int(40 * intv / 200.0)
        # convert_png2mp4(img_dir, mp4_path, fps, delete_imgdir=False)

        # if self.data_type == 'stream':
        #     mp4_path = os.path.join(out_dir, title+'c.mp4')
        #     convert_png2mp4(img_c_dir, mp4_path, fps, delete_imgdir=False)

        # self.sess.run(self.G_, {self.z: z_c[:self.b_num,:]})

        # st = time.time()
        # self.sess.run(self.G_, {self.z: z_c[:self.b_num,:]})
        # dt = time.time() - st

        # dt /= self.b_num
        # print('avg time: %f' % dt)
        # return
        
        G = None
        G_curl = None
        for b in range(niter):            
            G_ = self.sess.run(self.G_, {self.z: z_c[self.b_num*b:self.b_num*(b+1),:]})
            G_, _ = self.batch_manager.denorm(x=to_nhwc_numpy(G_))

            if G is None:
                G = G_
            else:
                G = np.concatenate((G, G_), axis=0)
            
            if self.data_type == 'stream':
                G_curl_ = self.sess.run(self.G_curl_, {self.z: z_c[self.b_num*b:self.b_num*(b+1),:]})
                G_curl_ = self.batch_manager.denorm_vel(x=to_nhwc_numpy(G_curl_))

                if G_curl is None:
                    G_curl = G_curl_
                else:
                    G_curl = np.concatenate((G_curl, G_curl_), axis=0)

        for i in range(intv):
            x = G[i]
            # img_path = os.path.join(img_dir, '%04d.png' % i)
            # save_image(x, img_path, single=True)
            
            if self.data_type == 'velocity':
                img_path = os.path.join(img_dir, 'w%04d.png' % i)
                xv = vortplot(x, img_path)
                # img_path = os.path.join(img_dir, 'j%04d.png' % i)
                # jacoplot(x, img_path)
            elif self.data_type == 'pressure' or\
                 self.data_type == 'stream':                
                img_path = os.path.join(img_dir, 'g%04d.png' % i)
                xg = gradplot(x, img_path)

            # if (title == 'intp_0_3_4' and i == 112) or\
            #    (title == '5_2_' and i == 96):
            #     img_path = os.path.join(img_dir, 's%04d.png' % i)
            #     xs = streamplot(x, img_path)
            #     # print(xs.shape, x.shape)
            #     # assert False

            #     import scipy.ndimage
            #     from PIL import Image
            #     xv = np.clip(scipy.ndimage.zoom(xv, [5,5,1], order=3), 0, 255)
            #     xv_ = xv[:,:,:3]/255.0
            #     xs_ = xs/255.0
            #     x = np.uint8(xv_*xs_*255)
            #     im = Image.fromarray(x)
            #     img_path = os.path.join(img_dir, 'sw%04d.png' % i)
            #     im.save(img_path)
            
        if dump:
            dump_path = os.path.join(out_dir, title+'.npz')
            np.savez_compressed(dump_path, v=G)

            if self.data_type == 'stream':
                dump_path = os.path.join(out_dir, title+'c.npz')
                np.savez_compressed(dump_path, v=G_curl)

    def generate(self, inputs, root_path=None, idx=None):
        generated = []
        for i, z_sample in enumerate(inputs):
            generated.append(self.sess.run(self.G, {self.y: z_sample}))
            
        c_concat = np.concatenate(tuple(generated[:-1]), axis=0)
        c_path = os.path.join(root_path, '{}_c.png'.format(idx))
        save_image(c_concat, c_path, nrow=self.b_num)
        print("[*] Samples saved: {}".format(c_path))

        if self.data_type == 'velocity':
            c_vort = self.get_vort_image(c_concat/127.5-1, is_vel=True)
            c_path = os.path.join(root_path, '{}_cv.png'.format(idx))
            save_image(c_vort, c_path, nrow=self.b_num)
            print("[*] Samples saved: {}".format(c_path))

        x = generated[-1]
        x_path = os.path.join(root_path, 'x_fixed_{}.png'.format(idx))
        if self.data_type == 'velocity':
            x_w = self.get_vort_image(x/127.5-1, is_vel=True)
            x_w = np.concatenate((x_w,x_w,x_w), axis=3)
            x = np.concatenate((x,x_w), axis=0)

        save_image(x, x_path)
        print("[*] Samples saved: {}".format(x_path))

    def to_vel(self, x):
        # normalized vel. by str. scale -> re-normalize vel. by velocity scale
        return self.batch_manager.to_vel(x)

    def get_curl_image(self, x):
        # x range [-1, 1], NHWC
        x = curl_np(x)
        x = self.to_vel(x)
        x_img = np.clip((x+1)*127.5, 0, 255)
        # print(x_img.shape)

        b_ch = np.ones((x_img.shape[0], self.height, self.width, 1))*127.5
        x_img = np.concatenate((x_img, b_ch), axis=-1)
        # print(x_img.shape)
        # x_img = np.reshape(x_img[:,:,:,0], (self.b_num, self.height, self.width, 1))
        return x_img

    def get_vort_image(self, x, is_vel=False):
        if is_vel:
            x = vort_np(x[:,:,:,:2])
        else:        
            # x range [-1, 1], NHWC
            x = vort_np(curl_np(x))
            x = self.to_vel(x)
        x_img = np.clip((x+1)*127.5, 0, 255)
        # print(x_img.shape)
        # x_img = np.reshape(x_img[:,:,:,0], (self.b_num, self.height, self.width, 1))
        return x_img

    def get_grad_image(self, x):
        x = grad_np(x)
        xr = [np.abs(x.min()), np.abs(x.max())]
        x[x<0] /= xr[0]
        x[x>0] /= xr[1]
        x_img = np.clip((x+1)*127.5, 0, 255)
        b_ch = np.ones((x_img.shape[0], self.height, self.width, 1))*127.5
        x_img = np.concatenate((x_img, b_ch), axis=-1)
        # x_img = np.uint8(plt.cm.RdBu(x_/255.0)*255)
        return x_img