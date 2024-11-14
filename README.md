# Semi-Supervised Normalizing Flow for Conditional Small Molecules Generation

## Installation

```
git clone git@github.com:sgalkina/semisupervised-molecules.git
cd semisupervised-molecules
conda create -n seminorm python=3.10
pip install -r requirements.txt
```

## Run on the HMDB dataset

With GPU (production):
```
python src/train.py trainer.max_epochs=200
```

With CPU (testing):
```
python src/train.py trainer=cpu
```