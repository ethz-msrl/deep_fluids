#! /usr/bin/env python

import os
import argparse
import tensorflow as tf
import shutil
from config import get_config
import numpy as np
import yaml
from model import GeneratorBE3

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts deep-fluids log folder to a saved model bundle that can be used in \
    mag_mag_manip')
    parser.add_argument('input_folder', help='path to a folder containing the tensorflow model checkpoints')
    parser.add_argument('model_name', help='the name of the output model')
    parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite the existing model with the same name')
    parser.add_argument('-z', '--zip', action='store_true', help='zip the output folder')

    args = parser.parse_args()

    # get the deep-fluids config
    config, unparsed = get_config()
    root = config.load_data_path        

    # read data generation arguments
    with open(os.path.join(root, 'args.txt'), 'r') as f:
        data_args = yaml.load(f)

    c_num = data_args['num_param']
    
    bbox = data_args['bbox']
    min_b = data_args['min_b']
    max_b = data_args['max_b']

    output_shape = (config.res_x, config.res_y, config.res_z, 3)

    z = tf.placeholder(dtype=tf.float32, shape=(1, c_num))
    G_, _ = GeneratorBE3(z, config.filters, output_shape,
                             num_conv=config.num_conv, repeat=config.repeat, reuse=False)
    graph = tf.Graph()
    model_fn = tf.train.latest_checkpoint(args.input_folder)

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, model_fn)
        #loader.restore(sess, model_fn) 
        inputs = {'input': tf.saved_model.utils.build_tensor_info(z)}
        outputs = {'output': tf.saved_model.utils.build_tensor_info(G_)}

        signature = tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs,
                outputs=outputs,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        model_dir = os.path.join('models', args.model_name)

        if args.overwrite:
            shutil.rmtree(model_dir)

        builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], 
                signature_def_map= {
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
                    },
                strip_default_attrs=True)
        builder.save()
        print('model is saved to', model_dir)

    if args.zip:
        shutil.make_archive(os.path.join('models', args.model_name), 'zip', model_dir)

    # export the params metadata file
    params = {
            'name': args.model_name,
            'num_coils': c_num, 
            # this output is hardcoded since there does not seem to be a good way to 
            # retrieve it automatically from the metagraph
            'signature_tag': tf.saved_model.tag_constants.SERVING,
            'input_op_name': z.name.split(':')[0],
            'output_op_name': G_.name.split(':')[0],
            'scaling_field': max(abs(min_b), abs(max_b)),
            'min_current': data_args['min_c'],
            'max_current': data_args['max_c'],
            'output_grid':
                {   
                    'dim_x': config.res_x,
                    'dim_y': config.res_y,
                    'dim_z': config.res_z,
                    'min_x': bbox[0],
                    'max_x': bbox[1],
                    'min_y': bbox[2],
                    'max_y': bbox[3],
                    'min_z': bbox[4],
                    'max_z': bbox[5],
                    }
            }

    with open(os.path.join(model_dir, 'params.yaml'), 'w') as f:
        f.write(yaml.dump(params))
  
