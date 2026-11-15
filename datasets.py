from torch.utils.data import Dataset
import numpy as np
import random
import cv2

def read_xrays(dir):
    xray = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
    xray = xray.astype(np.float32) / 255.0
    return xray

def read_masks(dir):
    mask = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.float32)
    return mask

def augment(xray, mask):
    # The X-ray and the mask must ALWAYS get the same change, or they stop matching.

    # 1) horizontal flip (50% chance)
    if random.random() < 0.5:
        xray = np.fliplr(xray)
        mask = np.fliplr(mask)

    # 2) vertical flip (50% chance)
    # Turned OFF: a real knee X-ray is never upside down. Remove the # to turn it on.
    # if random.random() < 0.5:
    #     xray = np.flipud(xray)
    #     mask = np.flipud(mask)

    # 3) small rotation, -10 to +10 degrees
    angle = random.uniform(-10, 10)
    h, w = xray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    xray = cv2.warpAffine(xray, M, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)  # NEAREST keeps mask 0/1

    # 4) random crop: cut out 85-100% of the picture, then resize back to full size (= small zoom)
    s = random.uniform(0.85, 1.0)
    ch, cw = int(h * s), int(w * s)
    top = random.randint(0, h - ch)
    left = random.randint(0, w - cw)
    xray = cv2.resize(xray[top:top + ch, left:left + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask[top:top + ch, left:left + cw], (w, h), interpolation=cv2.INTER_NEAREST)

    return xray, mask

class KneeDataset(Dataset):  # FIX: removed the extra space before "class"
    def __init__(self, df, augment=False):
        self.df = df
        self.augment = augment  # True only for training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        xray = read_xrays(self.df['xrays'].iloc[index])  # x value
        mask = read_masks(self.df['masks'].iloc[index])  # y value

        if self.augment:
            xray, mask = augment(xray, mask)

        res = {
            'xray': xray.reshape(1, *xray.shape).copy(),  # (H, W) -> (1, H, W)
            'mask': mask.reshape(1, *mask.shape).copy()
        }

        return res
