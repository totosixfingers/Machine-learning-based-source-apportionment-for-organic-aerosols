# **README.md — Machine Learning Source Apportionment for Organic Aerosols**

This repository implements several machine-learning–based source apportionment tools for **AMS (Aerosol Mass Spectrometer)** datasets, including:

* **Sparse Logistic Regression classifiers (L1, Elastic Net, XGBoost)**
    
* **Non-Negative Matrix Factorization (NMF)**
    
* **Linear Autoencoder (AE) decomposition**
    
* **Source-Based Autoencoder (SourceAE)** with fixed profiles, uncertainty estimation, and ME-2–style constraints
    
* **Residual diagnostics & correlation with ground truth**
    

The system is designed to reproduce and extend methods commonly used in aerosol science (PMF, ME-2, CMB).

* * *

# **Installation**

### Install dependencies

```bash
poetry install
```

### Enter the virtual environment

```bash
poetry shell
```

* * *

# **Input Data Requirements**

Measurement files must be given as **Excel (.xlsx)** with the following sheets:

| Sheet | Description |
| --- | --- |
| **X** or **measurements** | Time series spectra. First column = timestamps, remaining columns = m/z intensities. |
| **error** | Measurement uncertainty matrix σᵢⱼ (same shape as X). |
| **F** (optional) | Ground truth profiles for validation (m/z × k). |
| **G** (optional) | Ground truth contributions (time × k). |
| **info** (optional) | Metadata. |

Column names in sheet **X** must contain m/z values (e.g., `43`, `m/z 43`, `mz_43`, …).

* * *

# **1. Library Preparation Scripts**

## Convert training xlsx to CSV

```bash
python src/classifier/train_xlsx_2_csv.py \
    --input resources/library.xlsx \
    --output resources/library.csv \
    --sheet Sheet1
```

## Convert prediction xlsx to CSV

```bash
python src/classifier/predict_xlsx_2_csv.py \
    --input resources/RusanenEtAl_synthetic.xlsx \
    --output resources/RusanenEtAl_synthetic.csv \
    --sheet X
```

* * *

# **2. Train Classifiers (L1, ElasticNet, Adaptive Lasso, XGBoost)**

### Example — Train L1 Logistic Regression

```bash
python src/classifier/ams_sparse_logreg_train.py \
    --csv resources/library.csv \
    --outdir resources/ \
    --penalty l1 \
    --overwritemodel \
    --quiet
```

### Example — Elastic Net

```bash
python src/classifier/ams_sparse_logreg_train.py \
    --csv resources/library.csv \
    --outdir resources/ \
    --penalty elasticnet \
    --l1-ratios 0.5 0.75 0.9 1.0 \
    --overwritemodel \
    --quiet
```

* * *

# **3. Predict With Trained Classifiers**

```bash
python src/classifier/ams_sparse_logreg_predict.py \
    --csv resources/RusanenEtAl_synthetic.csv \
    --model resources/model_11_feat.joblib \
    --outdir resources/
```

* * *

# **4. Generate Normalized Class Fractions**

```bash
python src/classifier/create_fractions_csv.py \
    --xlsx resources/RusanenEtAl_synthetic.xlsx \
    --outdir resources/
```

* * *

# **5. Compare True vs Predicted Probabilities**

```bash
python src/classifier/prob_class_comparison.py \
    --true_csv resources/RusanenEtAl_synthetic_fractions.csv \
    --pred_csv resources/predictions_model_98feat_xgboost.csv \
    --outdir resources/comparison/ \
    --penalty xgboost
```

* * *

# **6. NMF Source Apportionment**

Run basic NMF decomposition:

```bash
python NMF.py \
    --input resources/RusanenEtAl_synthetic.xlsx \
    --output results/nmf_run \
    --k 5 \
    --max_iter 600
```

This produces:

* `*_F_learned.csv` — factor profiles
    
* `*_G_learned.csv` — contributions
    
* Profile & contribution plots
    
* Residual diagnostic plots
    
* Correlation with ground truth (if available)
    

* * *

# **7. Linear Autoencoder Decomposition (AE)**

```bash
python Train_AE.py \
    --input resources/RusanenEtAl_synthetic.xlsx \
    --output results/AE_run \
    --k 4 \
    --lr 1e-2 \
    --epochs 300
```

Features:

* Non-negativity enforced on encoder & decoder
    
* NNDSVDA initialization for stability
    
* Weighted loss using measurement uncertainty
    
* Automatic ground truth matching with Hungarian algorithm
    
* Residual diagnostics
    

* * *

# **8. Source-Based Autoencoder (SourceAE)**



```bash
python Train_AE.py \
    --input resources/RusanenEtAl_synthetic.xlsx \
    --output results/SourceAE_run \
    --k 4 \
    --epochs 300 \
    --fixed_profiles resources/library.xlsx \
    --fixed_labels HOA BBOA
```

### SourceAE Features

Hybrid decoder:  
`F = [F_fixed | F_free]`

Supports _multiple variants_ of fixed profiles from a library  
Each run automatically picks one variant per profile:

```
HOA_Chen_EUOverview_all  
BBOA_Chebaicheb_ATOLL  
...
```

Allows **uncertainty estimation** using:

* Multiple random fixed variants
    
* Multiple AE runs
    
* Variability of learned G and F
    

Automatic ground truth matching using optimal permutation.

 **normalization constraint**:  
Each factor profile is normalized via:

```
F[:, j] /= sum(F[:, j])
G[:, j] *= sum(F[:, j])
```

so that the reconstruction `X_hat = G Fᵀ` remains unchanged.

* * *

# **9. Multi-Run Uncertainty Estimation**

### Example — 10 runs with random fixed profiles:

```bash
python Train_AE.py \
    --input resources/RusanenEtAl_synthetic.xlsx \
    --output results/SourceAE_multi \
    --k 4 \
    --epochs 300 \
    --fixed_profiles resources/library.xlsx \
    --fixed_labels HOA BBOA \
    --n_runs_sourceae 10 \
    --random_fixed_profiles
```

Outputs:

* Mean & std of **F** and **G**
    
* Per–component variability plots
    
* Uncertainty histograms
    
* Stacked contributions with uncertainty shading
    

* * *

# **10. Residual Diagnostics**

Each model produces:

###  Scaled residuals vs m/z

Mean ± standard deviation across time.

###  Scaled residuals vs time

Daily structure & model drift.

###  Full residual distribution

Histogram + normal fit + KDE.

* * *

# **11. Model Theory Summary**

### **Autoencoder decomposition**

A linear AE performs:

```
G = Encoder(X)
X_hat = G @ Fᵀ
```

Equivalent to constrained NMF when non-negativity is enforced.

### **SourceAE**

Introduces fixed profiles:

```
F = [F_fixed | F_free]
X_hat = G @ Fᵀ
```


### **Loss function (Q)**

Weighted by measurement error:

```
Q = mean( ((X - X_hat) / σ)² )
```


* * *

# **12. Exit Environment**

```bash
exit
```

* * *