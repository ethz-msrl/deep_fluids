import argparse
import os
import numpy as np
from util import create_test_train_indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='idx')
    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    from config import get_config
    config, unparsed = get_config()

    cmag_path = os.path.join(config.load_data_path, 'master_feature_matrix_v3.npy')
    data = np.load(cmag_path)
    nrow = 119
    n = data.shape[0] // nrow 

    # remove 10 % for testing
    if not args.no_shuffle:
        print('shuffling data with seed %s' % args.seed)
    idx_train, idx_test = create_test_train_indices(n, test_size=args.test_ratio, 
            shuffle=(not args.no_shuffle), seed=args.seed)
    print('training with %d samples testing with %d' % (len(idx_train), len(idx_test)))

    np.savetxt(os.path.join(config.load_data_path, args.prefix + '_train.txt'), idx_train, fmt='%d')
    np.savetxt(os.path.join(config.load_data_path, args.prefix + '_test.txt'), idx_test, fmt='%d')
