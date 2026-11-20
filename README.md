# Knee X-ray Segmentation with U-Net

Final project for Machine Vision and CNNs.
A U-Net finds the knee bone (tibia) in 224x224 grayscale X-rays.
## Model and license
- Model: U-Net (UNetLext in model.py), trained from scratch on our own knee X-rays.
  No pretrained weights were used.
- Code source: model.py comes from the rocaseg package by MIPT-Oulu (University of Oulu),
  written by Aleksei Tiulpin (2017):
  https://github.com/MIPT-Oulu/RobustCartilageSegmentation (file rocaseg/models/unet_lext.py).
  It is released under the MIT License (as stated in that project's setup.py).
  I changed only small things (default width/depth, x.float(), and the warning).
- U-Net architecture: Ronneberger, Fischer & Brox, MICCAI 2015.
- Libraries: PyTorch (BSD-3-Clause), OpenCV (Apache 2.0), pandas (BSD-3-Clause), matplotlib.

## Results
| | Dice |
|---|---|
| Validation (20 images) | ~0.978 |
| **Test (50 unseen images)** | **0.980 ± 0.002** (mean of several runs) |

## Files
| File | What it does |
|---|---|
| `data_preparation.py` | Makes train / val / test CSV lists |
| `datasets.py` | Loads X-rays + masks, augmentation (flip, rotation, random crop) |
| `model.py` | U-Net (depth 5) |
| `trainer.py` | Training loop (BCE + Dice loss), saves model with lowest val loss |
| `evaluate.py` | Loss/Dice graphs and prediction pictures |
| `utils.py` | Dice loss and Dice score |
| `args.py` | Settings (epochs, batch size, learning rate) |
| `main.py` | Runs everything |

## How to run
First unzip `data.zip` here, so you get a `data/` folder (xrays, masks, test_masks).
```
pip install torch torchvision opencv-python pandas matplotlib
python data_preparation.py
python main.py
```
Results (model, graphs, predictions) are saved in `sessions/`.
