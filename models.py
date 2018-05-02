import numpy as np
import tensorflow as tf
from ops import *
slim = tf.contrib.slim

def GeneratorBE(z, filters, output_shape, name='G',
                num_conv=3, conv_k=3, last_k=5, skip_conn=True, act=lrelu, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as vs:
        repeat_num = int(np.log2(output_shape[0])) - 2
        x0_shape = np.power(2, np.log2(output_shape[:2]) - (repeat_num - 1)).tolist()
        x0_shape = [int(i) for i in x0_shape] + [filters]
        # print(x0_shape, output_shape)
        num_output = int(np.prod(x0_shape))
        x = linear(z, num_output)
        x = reshape(x, x0_shape[0], x0_shape[1], x0_shape[2])
        if skip_conn: x0 = x
        
        for idx in range(repeat_num):
            for _ in range(num_conv):
                x = conv2d(x, filters, k=conv_k, s=1, act=act)

            if idx < repeat_num - 1:
                x = upscale(x, 2)
                # print('x shape', x.get_shape())
                if skip_conn:
                    x0 = upscale(x0, 2)
                    # print('x0 shape', x0.get_shape())
                    x = tf.concat([x, x0], axis=-1)

        out = conv2d(x, output_shape[-1], k=last_k, s=1)
        # out = tf.clip_by_value(out, -1, 1)

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
    b_num = 8
    res_y = 128
    res_x = 96
    ch_num = 2

    filters = 128
    c_num = 3

    x = tf.placeholder(dtype=tf.float32, shape=[b_num, res_y, res_x, ch_num])
    z = tf.placeholder(dtype=tf.float32, shape=[b_num, c_num])
    output_shape = get_conv_shape(x)[1:]

    dec, d_var = GeneratorBE(z, filters, output_shape, name='dec')
    show_all_variables()
    # tf.reset_default_graph()
    # return

if __name__ == '__main__':
    tf.app.run()
