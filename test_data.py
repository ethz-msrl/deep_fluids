import tensorflow as tf
from config import get_config
from data import BatchManager
from util import prepare_dirs_and_logger
from ops import get_conv_shape
from model import GeneratorBE3

if __name__ == '__main__':
    config, _ = get_config()
    prepare_dirs_and_logger(config)
    batch_manager = BatchManager(config)
    x, y = batch_manager.batch()
    output_shape = get_conv_shape(x)[1:]
    z = tf.placeholder(dtype=tf.float32, shape=[8, 8])
    G = GeneratorBE3(z, config.filters, output_shape, reuse=False)
    print G

    
