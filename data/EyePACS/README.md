# EyePACS

Obtain the source training images and labels from the
[Diabetic Retinopathy Detection competition](https://www.kaggle.com/c/diabetic-retinopathy-detection).
Kaggle access and acceptance of its terms may be required.

Expected layout relative to this directory:
```text
labels/trainLabels.csv
train_images/train/<image>.jpeg
```

EyePACS is used for source training and validation. APTOS supplies the separate
target-domain sets. See [data preparation](../../experiments/README.md#dataset-folders).
