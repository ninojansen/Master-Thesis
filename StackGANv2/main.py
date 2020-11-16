import argparse
import yaml
from dotmap import DotMap
import tensorflow as tf
from trainer import StackGANv2Trainer
from dataset import Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='StackGANv2 Tensorflow implementation')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='flowers_cfg.yml', type=str)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('-hp', dest='hide_progress', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    with open(args.cfg_file, 'r') as stream:
        cfg = DotMap(yaml.safe_load(stream))

    dataset = Dataset(batch_size=cfg.TRAIN.BATCH_SIZE)
    dataset.load_flowers()

    trainer = StackGANv2Trainer(cfg, dataset)

    trainer.train(hide_progress=args.hide_progress)

    print("Finished!")
