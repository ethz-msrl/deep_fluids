import os
import numpy as np
from config import get_config
from main import main
from tester import Tester
from util import prepare_dirs_and_logger
import tensorflow as tf

cmag_dir = os.path.join('data', 'cmag_dataset')
idx_train = np.loadtxt(os.path.join(cmag_dir, 'idx_train.txt'), dtype=np.int)

splits = np.arange(0.1, 1.0, 0.1)

for split in splits:
    n_subset = int(split * len(idx_train))
    idx_subset = idx_train[0:n_subset] 
    fn = os.path.join(cmag_dir, 'idx_train_ss_%d.txt' % (100 * split))
    np.savetxt(fn, idx_subset, fmt='%d')

    config, unparset = get_config()

    config.train_idx = os.path.basename(fn)
    config.tag = ('ss_%d' % (100 * split))
    print('training with %d %% of the training set' % (100 * split))
    main(config)

    # we need to clear the TF session between training and testing
    tf.reset_default_graph()

    config.load_model_dir = config.model_dir
    print('testing with %d %% of the training set' % (100 * split))
    tester = Tester(config)
    tester.test()

    # we need to clear the TF session between training and testing
    tf.reset_default_graph()
