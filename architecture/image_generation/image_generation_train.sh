#!/bin/bash
#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16G
#SBATCH --job-name=IG

module purge
module load scikit-image/0.16.2-fosscuda-2019b-Python-3.7.4 
module load CUDA/10.1.243-GCC-8.3.0

python main.py --cfg cfg/easyVQA/default.yml --ef_type $ef --type all --check_val_every_n_epoch 1 --data_dir /data/s2965690/datasets/ExtEasyVQA/ --outdir /data/s2965690/$name --progress_bar_refresh_rate 0
# --max_epochs 1 --limit_train_batch 0.01 --limit_val_batch 0.1 
# --data_dir /data/s2965690/datasets/ExtEasyVQA/ --outdir /data/s2965690/$name --progress_bar_refresh_rate 0





