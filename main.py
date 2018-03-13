import numpy as np
import tensorflow as tf

from data import BatchManager
from config import get_config
from utils import prepare_dirs_and_logger, save_config

def main(config):
    if config.archi == 'de':
        from trainer import Trainer
    elif config.archi == 'dg':
        from trainer_dg import TrainerDG
        Trainer = TrainerDG
    else:
        raise Exception("[!] You should specify `archi` to load a trainer")

    prepare_dirs_and_logger(config)
    tf.set_random_seed(config.random_seed)

    batch_manager = BatchManager(config)
    trainer = Trainer(config, batch_manager)

    if config.is_train:
        save_config(config)
        trainer.train()
    else:
        if not config.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer.test()

if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
