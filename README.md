# GAN-based Imputation of Time Series

### Quickstart

Running the following command will download UCIHAR in ``'./data'`` directory, 
generate missing sequences corresponding to 50% of missingness, train a GAN then impute
the missing data. And finally a GRU classifier is trained on the imputed data and the results
are saved as ``tensorboard`` event files in ``'./runs/UCIHAR'`` director.


```python impute_classify --root_dir=data --miss_rate=0.5 --log_dir=runs```