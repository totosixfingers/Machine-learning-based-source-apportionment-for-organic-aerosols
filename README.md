# Machine Learning Source Apportionment for Organic Aerosols

This project allows training and applying ML models to apportion organic aerosol sources using AMS (Aerosol Mass Spectrometer) data.

---

##### Install dependencies

```
$ poetry install
```

##### Start a shell within the virtual environment

```
$ poetry shell
```

-----------

#### <u>Prepare data</u>

##### Transform training data from xlsx to csv

```
$ python src/classifier/train_xlsx_2_csv.py --input resources/library.xlsx --output resources/library.csv --sheet Sheet1
```

##### Transform prediction data from xlsx to csv 

```
$ python src/classifier/predict_xlsx_2_csv.py --input resources/RusanenEtAl_synthetic.xlsx --output resources/RusanenEtAl_synthetic.csv --sheet X
```

------------------

#### <u>Train the model</u>

```
$ python src/classifier/ams_sparse_logreg_train.py --csv <train_data>.csv --outdir <folder_name> --features <ints...> --penalty {l1, elasticnet} --quiet --overwritemodel
```

# Training Script Arguments

- `--csv <train_data>.csv`  
  Path to the CSV file containing spectra and `label` column.

- `--outdir <output_folder>`  
  Directory to save the trained model. Created if it doesn't exist.

- `--penalty {l1, elasticnet, l2, adaptive, xgboost}`  
  Type of model to train:  
  - `l1` → Lasso logistic regression  
  - `elasticnet` → L1 + L2 mixture  
  - `l2` → Ridge logistic regression  
  - `adaptive` → Adaptive Lasso (two-step weighted L1)  
  - `xgboost` → Tree-based gradient boosting

- `--l1-ratios <floats...>` *(required if `--penalty elasticnet`)*  
  Grid of L1 ratios to try (1.0 = pure L1). Example: `--l1-ratios 0.5 0.75 0.9 1.0`

- `--overwritemodel`  
  Overwrite an existing model file without asking.

- `--quiet`  
  Suppress convergence warnings from scikit-learn.

  ## Examples

### 1. L1 Logistic Regression
```
$ python src/classifier/ams_sparse_logreg_train.py --csv resources/library.csv --outdir resources/  --penalty l1 --quiet  --overwritemodel
```

```
$ python src/classifier/ams_sparse_logreg_train.py --csv resources/library.csv --outdir resources/ --penalty elasticnet --l1-ratios 0.5 0.75 0.9 1.0  --quiet \--overwritemodel
```

  --------------------

#### <u>Predict with the generated model</u>

```
$ python src/classifier/ams_sparse_logreg_predict.py --csv resources/RusanenEtAl_synthetic.csv --model resources/model_11_feat.joblib --outdir resources/ 
```
On resources see the name of the file you produced.
----------------

##### Exit the poetry shell

```
$ exit
```
