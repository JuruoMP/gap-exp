# download sparc dataset
pip install gdown
gdown --id 13Abvu5SUMSP3SJM-ZIj66mOkeyAquR73
unzip sparc.zip

# download pre-trained checkpoint
curl https://gap-text2sql-public.s3.amazonaws.com/checkpoint-artifacts/pretrained-checkpoint -o pretrained_checkpoint/pytorch_model.bin

# run
python run.py train experiments/sparc-contrast-configs/gap-run.jsonnet