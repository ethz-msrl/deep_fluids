import os
import numpy as np
from config import get_config
from main import main
from tester import Tester
from util import prepare_dirs_and_logger
import tensorflow as tf

cmag_dir = os.path.join('data', 'cmag_dataset')
idx_train = np.loadtxt(os.path.join(cmag_dir, 'idx_train.txt'), dtype=np.int)

epochs = np.array([50, 100, 150, 200, 250, 300, 350, 400])
#epochs = np.array([300, 350, 400])

for epoch in epochs:
    config, unparset = get_config()

    config.max_epoch = epoch
    config.tag = ('ep_%d' % (epoch))
    config.use_curl = False
    print('training with %d epochs' % epoch)
    main(config)

    # we need to clear the TF session between training and testing
    tf.reset_default_graph()

    config.load_model_dir = config.model_dir
    print('testing with %d epochs' % (epoch))
    tester = Tester(config)
    tester.test()

    # we need to clear the TF session between training and testing
    tf.reset_default_graph()
