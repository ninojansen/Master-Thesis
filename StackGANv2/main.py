import argparse
import yaml
from dotmap import DotMap
import tensorflow as tf
from trainer import StackGANv2Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='StackGANv2 Tensorflow implementation')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cifar10_cfg.yml', type=str)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    args = parser.parse_args()
    return args


def prepare_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    train_images = (train_images - 127.5) / 127.5

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    train_labels_encoded = tf.keras.utils.to_categorical(train_labels)

    return tf.data.Dataset.from_tensor_slices((train_images, train_labels, train_labels_encoded))


if __name__ == "__main__":
    args = parse_args()

    with open(args.cfg_file, 'r') as stream:
        cfg = DotMap(yaml.safe_load(stream))

    if cfg.DATASET_NAME == "cifar-10":
        dataset = prepare_data()
    else:
        dataset = None

    trainer = StackGANv2Trainer(cfg, dataset)

    # trainer.train()

    print("Finished!")
