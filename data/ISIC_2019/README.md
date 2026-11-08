# ISIC 2019

Obtain images, metadata, and ground-truth labels from the
[ISIC Challenge data portal](https://challenge.isic-archive.com/data/),
selecting the 2019 dataset and following its access terms.

Expected layout relative to this directory:
```text
ISIC_2019_Training_GroundTruth.csv
ISIC_2019_Training_Metadata.csv
ISIC_2019_Test_GroundTruth.csv
train/ISIC_2019_Training_Input/<image>.jpg
test/ISIC_2019_Test_Input/<image>.jpg
```

The labeled official test set is required for the reported evaluation.
Unknown-class test observations are exported separately. See
[data preparation](../../experiments/README.md#dataset-folders).
