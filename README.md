# unsupervised-domain-adaptation
Deep learning project code record
Please make sure you have the following libraries installed:
- numpy
- torch>=1.4.0
- torchvision>=0.5.0

## Datasets
- [Office-31](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)
- [Office-Home](https://www.hemanthdv.org/OfficeHome-Dataset/)
- [DomainNet](http://ai.bu.edu/M3SDA/) 

We provide direct download links in the script. However, for file larger than 100 MB (OfficeHome - Art and RealWorld), please manually download them from the following links--[Art](https://drive.google.com/file/d/18_F4TMEwP3yJcxTwhgH3FFc7OYtSJxAj/view?usp=sharing) and [RealWorld](https://drive.google.com/file/d/1xZPldApUAkx4qzsRIi00qfnzeX13HWCe/view?usp=sharing)--and extract them appropriately.

## Usage
- `beta` - learning rate/ momentum parameter to learn proportions in the target domain ( `beta=0` corresponds to using a uniform prior)
- `sub_s` - subsample the source dataset
- `sub_t` - subsample the target dataset

Below, we provide example commands to run our method.
```shell script
# Train PCT on Office-31 Amazon -> Webcam task using ResNet 50.
# Assume you have put the datasets under the path `data/office-31

# Single-source adaptation 
python examples/proto.py data/office31 -d Office31 -s A -t W -a resnet50  --epochs 10

# Sub-sampled source adaptation (uniform prior)
python examples/proto.py data/office31 -d Office31 -s A -t W -a resnet50  --epochs 10 --sub_s

# Sub-sampled source adaptation (learnable prior)
python examples/proto.py data/office31 -d Office31 -s A -t W -a resnet50  --epochs 10 --sub_s --beta 0.001

# Sub-sampled target adaptation (uniform prior)
python examples/proto.py data/office31 -d Office31 -s A -t W -a resnet50  --epochs 10 --sub_t

# Sub-sampled target adaptation (learnable prior)
python examples/proto.py data/office31 -d Office31 -s A -t W -a resnet50  --epochs 10 --sub_t --beta 0.001

```
Example commands are included in examples/proto.sh.

For source-private adaptation, please follow the instruction in the readme.md in the `Proto_Private' folder.
