import os
import pandas as pd
import matplotlib.pyplot as plt

data_list = []
xrays_dir = "data/xrays"
csv_dir = "data/CSVs"
os.makedirs(csv_dir, exist_ok=True)

files = [f for f in os.listdir(xrays_dir) if f.endswith(".png")]
for xrays in sorted(files, key=lambda f: int(os.path.splitext(f)[0])):
    xray_path = os.path.join(xrays_dir, xrays)
    mask_path = xray_path.replace("xrays", "masks")

    if not os.path.exists(mask_path):
        mask_path = None

    data_list.append((xray_path, mask_path))

data_df = pd.DataFrame(data_list, columns=["xrays", "masks"])
data_df.to_csv(os.path.join(csv_dir, "dataset.csv"), index=False)

#--------------"train,test,val"--------------------------------------------#
data_df = pd.read_csv(os.path.join(csv_dir, "dataset.csv"))

test_df = data_df[data_df["masks"].isna()]
labeled_df = data_df[data_df["masks"].notna()]

# NEW: the test masks live in their own folder (data/test_masks), so they never end up in train/val
test_df = test_df.copy()
test_df["masks"] = test_df["xrays"].str.replace("xrays", "test_masks")

val_df = labeled_df.sample(frac=0.2, random_state=42)
train_df = labeled_df.drop(val_df.index)

train_df.to_csv(os.path.join(csv_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(csv_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(csv_dir, "test.csv"), index=False)

#--------------"graph"--------------------------------------------#
names = ["train", "val", "test"]
counts = [len(pd.read_csv(os.path.join(csv_dir, f"{n}.csv"))) for n in names]

plt.bar(names, counts)
for i, c in enumerate(counts):
    plt.text(i, c, str(c), ha="center", va="bottom")
plt.title("Number of samples per split")
plt.ylabel("Samples")
plt.show()