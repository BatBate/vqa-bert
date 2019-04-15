# vqa-bert

### Installingenvironment

1. Install Anaconda (Anaconda recommended: https://www.continuum.io/downloads).
2. Install cudnn v7.0 and cuda.9.0
3. Create environment for pythia
```bash
conda create --name vqa python=3.6

source activate vqa
pip install demjson pyyaml

pip install torch==0.4.1

pip install torchvision
pip install tensorboardX
pip install pytorch-pretrained-bert

```
