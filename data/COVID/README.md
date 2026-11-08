# COVID-19 Radiography Database

Obtain the data from the provider's
[COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
and follow its access and usage terms.

Expected layout relative to this directory:
```text
Normal/images/*.png
COVID/images/*.png
Viral Pneumonia/images/*.png
Lung_Opacity/images/*.png
```

The preparation script reads images, not masks. See
[data preparation](../../experiments/README.md#dataset-folders).
