module load cuda/11.8
module load gcc/11.2.0
module load anaconda3/2023.03-py3.10

source ~/.bashrc
conda activate seminorm

/home/nmb127/.conda/envs/seminorm/bin/python src/train.py trainer.max_epochs=200 data.n_workers=27
