import argparse
from StackGAN import StackGAN


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='StackGAN.')
    parser.add_argument('--cfg', default="cfg.yml")
    parser.add_argument('--data_dir', default="/home/nino/Documents/Datasets/Birds")
    parser.add_argument('--output_dir', default="./output")
    parser.add_argument('-display_progress', action='store_true', default=False)
    parser.add_argument('-small', action='store_true', default=False)
    args = parser.parse_args()
    stackGAN = StackGAN(cfg_file=args.cfg, data_dir=args.data_dir, output_dir=args.output_dir)
    stackGAN.train(display_progress=args.display_progress)
