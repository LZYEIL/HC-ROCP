# APTOS 2019

Obtain the labeled training images and labels from the
[APTOS 2019 Blindness Detection competition](https://www.kaggle.com/c/aptos2019-blindness-detection).
Kaggle access and acceptance of its terms may be required.

Expected files relative to this directory:
```text
train.csv
train_images/<id_code>.png
```

The preparation script splits these labeled images into target calibration,
development, and test sets. The unlabeled competition test images are not
required for the reported experiments. See [data preparation](../../experiments/README.md#dataset-folders).
