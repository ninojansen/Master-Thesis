import argparse
import yaml
from dotmap import DotMap
import tensorflow as tf
from trainer import StackGANv2Trainer
from dataset import Dataset

# TODO
# Consider using pretraining with VAE https://arxiv.org/pdf/2002.02112.pdf
# look into tensorboard viz
# Add CA embedding and kl loss


def parse_args():
    parser = argparse.ArgumentParser(description='StackGANv2 Tensorflow implementation')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='./config/flowers_cfg.yml', type=str)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('-hp', dest='hide_progress', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    with open(args.cfg_file, 'r') as stream:
        train_cfg = DotMap(yaml.safe_load(stream))

    with open("./config/main_cfg.yml", 'r') as stream:
        main_cfg = DotMap(yaml.safe_load(stream))
    dataset = Dataset(batch_size=train_cfg.TRAIN.BATCH_SIZE)
    dataset.load_flowers()

    trainer = StackGANv2Trainer(main_cfg, train_cfg, dataset)

    trainer.train(hide_progress=args.hide_progress)

    print("Finished!")
