import numpy as np
import tensorflow as tf
from ops import *

def GeneratorBE(z, filters, output_shape, name='G',
                num_conv=4, conv_k=3, last_k=3, repeat=0, skip_concat=False, act=lrelu, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as vs:
        # repeat_num = int(np.log2(output_shape[0])) - 2
        # assert(repeat_num > 0)
        # x0_shape = np.power(2, np.log2(output_shape[:-1])-repeat_num+1).tolist()
        # x0_shape = [int(np.round(i)) for i in x0_shape] + [filters]
        # os = [i*np.power(2, repeat_num-1) for i in x0_shape[:-1]]
        # assert(os[0] == output_shape[0] and os[1] == output_shape[1])
        if repeat == 0:
            repeat_num = int(np.log2(np.max(output_shape[:-1]))) - 2
        else:
            repeat_num = repeat
        assert(repeat_num > 0 and np.sum([i % np.power(2, repeat_num-1) for i in output_shape[:-1]]) == 0)

        x0_shape = [int(i/np.power(2, repeat_num-1)) for i in output_shape[:-1]] + [filters]
        print('first layer:', x0_shape, 'to', output_shape)

        num_output = int(np.prod(x0_shape))
        layer_num = 0
        x = linear(z, num_output, name=str(layer_num)+'_fc')
        layer_num += 1
        x = reshape(x, x0_shape[0], x0_shape[1], x0_shape[2])
        x0 = x
        
        for idx in range(repeat_num):
            for _ in range(num_conv):
                x = conv2d(x, filters, k=conv_k, s=1, act=act, name=str(layer_num)+'_conv')
                layer_num += 1

            if idx < repeat_num - 1:
                if skip_concat:
                    x = upscale(x, 2)
                    x0 = upscale(x0, 2)
                    x = tf.concat([x, x0], axis=-1)
                else:
                    x += x0
                    x = upscale(x, 2)
                    x0 = x

            elif not skip_concat:
                x += x0
        
        out = conv2d(x, output_shape[-1], k=last_k, s=1, name=str(layer_num)+'_conv')
        # out = tf.clip_by_value(out, -1, 1)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables

def GeneratorBE3(z, filters, output_shape, name='G',
                num_conv=4, conv_k=3, last_k=3, repeat=0, skip_concat=False, act=lrelu, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as vs:
        if repeat == 0:
            repeat_num = int(np.log2(np.max(output_shape[:-1]))) - 2
        else:
            repeat_num = repeat
        assert(repeat_num > 0 and np.sum([i % np.power(2, repeat_num-1) for i in output_shape[:-1]]) == 0)
        x0_shape = [int(i/np.power(2, repeat_num-1)) for i in output_shape[:-1]] + [filters]
        print('first layer:', x0_shape, 'to', output_shape)

        num_output = int(np.prod(x0_shape))
        layer_num = 0
        x = linear(z, num_output, name=str(layer_num)+'_fc')
        layer_num += 1
        x = tf.reshape(x, [-1] + x0_shape)
        x0 = x
        
        for idx in range(repeat_num):
            for _ in range(num_conv):
                x = conv3d(x, filters, k=conv_k, s=1, act=act, name=str(layer_num)+'_conv')
                layer_num += 1

            if idx < repeat_num - 1:
                if skip_concat:
                    x = upscale3(x, 2)
                    x0 = upscale3(x0, 2)
                    x = tf.concat([x, x0], axis=-1)
                else:
                    x += x0
                    x = upscale3(x, 2)
                    x0 = x

            elif not skip_concat:
                x += x0

        out = conv3d(x, output_shape[-1], k=last_k, s=1, name=str(layer_num)+'_conv')

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables

def DiscriminatorPatch(x, filters, name='D', train=True, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as vs:
        repeat_num = 3 # if c4k3s2, rfs 95, w/16=8, if c3k3s2, rfs 47, w/8=16
        d = int(filters/2)
        for _ in range(repeat_num): 
            x = conv2d(x, d, k=3, act=lrelu) # 64/32/16-64/128/256
            d *= 2
        x = conv2d(x, d, k=3, s=1, act=lrelu) # 16x16x512
        out = conv2d(x, 1, k=3, s=1) # 16x16x1

        # x = conv2d(x, int(d/2), k=3, s=2, act=lrelu) # 8x8x256
        # b = get_conv_shape(x)[0]
        # flat = tf.reshape(x, [b, -1])
    variables = tf.contrib.framework.get_variables(vs)    
    return out, None, variables

def main(_):
    # #########
    # 2d
    b_num = 8
    res_y = 128
    res_x = 96
    # res_y = 64
    # res_x = 128
    # res_y = 240
    # res_x = 480
    ch_num = 2

    filters = 128
    c_num = 3

    x = tf.placeholder(dtype=tf.float32, shape=[b_num, res_y, res_x, ch_num])
    z = tf.placeholder(dtype=tf.float32, shape=[b_num, c_num])
    output_shape = get_conv_shape(x)[1:]

    dec, d_var = GeneratorBE(z, filters, output_shape, name='dec')
    ##############
   
    # #########
    # # 3d
    # b_num = 4
    # res_x = 112
    # res_y = 64
    # res_z = 32
    # ch_num = 3

    # filters = 128
    # c_num = 3

    # x = tf.placeholder(dtype=tf.float32, shape=[b_num, res_z, res_y, res_x, ch_num])
    # z = tf.placeholder(dtype=tf.float32, shape=[b_num, c_num])
    # output_shape = int_shape(x)[1:]

    # dec, d_var = GeneratorBE3(z, filters, output_shape, name='dec')
    ##############


    show_all_variables()
    # tf.reset_default_graph()
    # return

if __name__ == '__main__':
    tf.app.run()
