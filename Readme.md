# Case_Study_Zurich_Insurance_Grey
A case study made for insurance claims with Zurich Insurance group

# Risk Scorer (Commercial Auto Claims Prediction)

This repository contains the end-to-end predictive modeling pipeline to score commercial auto insurance policies for expected claim risk. It trains classification models on historical policy data (2020–2022) to predict the probability of a claim occurring during the current period (2023–2024).

---

The output is a prediction.csv which models the risk factor.

## Summary
01 | Exploratory Data Analysis (EDA)
   • Diagnosed high class imbalance (~12.35% claims) and categorical text inconsistencies.
   • Identified 5 major target-leak variables (e.g., claim status, loss amounts).

02 | Data Refining & Multi-Paradigm Setup
   • Pruned Complete: Zero-noise, lower volume.
   • Artificially Completed: Mean/mode imputation.
   • Sparse: Kept raw NaNs intact to leverage native HGB splitting.

03 | Robust Engineering & Validation
   • Built safe preprocessing pipelines to drop leakage and handle unseen categoricals.
   • Designed a Temporal Validation Split (Train: 2020-21, Val: 2022) to mimic production.

04 | Performance Benchmarking & Interpretability
   • Evaluated 3 models. Resolved the "accuracy-based" feature importance bottleneck.
   • Unlocked sensitive permutation importances using ROC-AUC (Top drivers: Coverage & Fleet size).

05 | Production Deployment Roadmap
   • Plan for modularizing into standard python packages, automated CI/CD, and drift monitoring.



## File System
```text
├── case study explanation.docx              # Business context and task guidelines
├── Data Scientist Applied AI ML - Case Study.docx # Official project case study brief
├── Readme.md                                # Full documentation (system architecture & setup)
├── risk_scorer.ipynb                        # Complete modeling notebook (EDA to Export)
├── predictions.csv                          # WINNING MODEL PREDICTIONS (Copy of predictions_sparse.csv)
│
├── data/                                    # Input Data Folder
│   ├── data_dictionary.csv                  # Schema & definitions for all 31 variables
│   ├── train.csv                            # Historical training policies (2020-22 with claim outcomes)
│   └── score.csv                            # Unlabeled scoring policies (2023-24 to be scored)
│
├── data_edits/                              # Intermediate Datasets
│   ├── fixed_train.csv                      # Imputed training dataset (using mean/mode defaults)
│   └── train_pruned.csv                     # Pruned training dataset (incomplete rows removed)
│
└── outputs/                                 # Exported Model Artifacts
    ├── predictions_sparse.csv               # Raw winning predictions (HistGradientBoosting)
    ├── predictions_imputed.csv              # Experimental predictions (Imputed Random Forest)
    ├── predictions_pruned.csv               # Experimental predictions (Pruned Random Forest)
    └── model_feature_importances.csv        # Comprehensive 31-column feature weights matrix
```


## 1. Exploratory Data Analysis (EDA)

During initial exploration of the raw `train.csv` dataset, we identified several patterns, anomalies, and structural surprises that heavily informed our design:

### Data Quality & Missingness Patterns

Out of $15,075$ historical training records, several critical variables contained missing entries:

* **Continuous Features:** `driver_avg_age` (603 missing), `prior_year_mileage_000` (903 missing), and `risk_score_external` (752 missing). Directing these to a default value of $0$ would introduce significant negative bias.


* **Behavioral Features:** `late_payment_count` (1,053 missing) and `prior_loss_amount` (1,208 missing).


* **Text Categorical Formatting:** `business_type` contained inconsistent capitalization (e.g., `"service"`, `"Service"`, and `"SERVICE"`).
* **Structural Leakage Variables:** `first_claim_reported_date` and `days_to_first_claim_report` had exactly $13,214$ missing records. This matches the number of zero-claim policies, confirming these variables are populated *post-hoc* (after a claim is filed).



### Imbalanced Target Distribution

The historical target rate is highly imbalanced:

* **No Claim ($0$):** $13,214$ policies.


* **Claim ($1+$):** $1,861$ policies (~12.35% baseline positive rate).
Because underwriters care about actual risk probabilities for tier-pricing rather than simple binary classifications, we focused on probability modeling assessed by **ROC AUC** and **Gini index** rather than standard accuracy.



---

## 2. Modeling Methodology Improvements

To maximize predictive performance on unseen data, we implemented three key methodological improvements:

### 1. Robust Leakage and Metadata Filtering

We stripped out "time-machine" variables that create artificial accuracy but are impossible to observe at underwriting time:

* **Target Leaks:** `claim_paid_amount_current_period`, `days_to_first_claim_report`, `first_claim_reported_date`, `claim_status_current_period`, and `total_loss_amount`.


* **ID/Metadata:** `policy_id`, `insured_id`, `snapshot_date`, `policy_effective_date`, and `policy_expiration_date`.



### 2. Multi-Paradigm Data Exploration

Instead of relying on a single data imputation strategy, we processed and evaluated three distinct data philosophies:

1. **Pruned Complete:** Dropping any row containing a missing predictor.


2. **Artificially Completed (Imputed):** Filling missing numbers with column means, missing categoricals with modes, and resolving context-specific fields (e.g., setting `prior_loss_amount` to $0$ if there were no prior claims).


3. **Sparse (NaNs Intact):** Keeping missing values completely intact and leveraging advanced algorithms that handle sparsity natively.



### 3. Rigorous Out-of-Sample Validation

Instead of a random train-test split, we built a custom **Temporal Validation Split**.

* **Train Set:** Policies effective in **2020 and 2021**.


* **Validation Set:** Policies effective in **2022** (the most recent historical year).
This closely simulates how the model performs in production when scoring forward-looking portfolios (`score.csv` spanning 2023–2024).



---

## 3. Production Readiness Improvements

We implemented three high-impact design changes to transition this from a fragile notebook to a deployable, production-ready system:

### 1. Self-Contained Preprocessing & Inference Alignment

The `preprocess_datasets` function decouples training and scoring files. It calculates statistical metadata (means/modes) from the training data and caches them to impute the scoring set at inference time. This guarantees that if a variable is missing in `score.csv`, it is filled using stable training baseline statistics—preventing out-of-bounds runtime calculation crashes.

### 2. Standardized Categorical Pipeline

Categoricals are passed through a strict text normalization filter (removing whitespaces and forcing uppercase) before being mapped with an `OrdinalEncoder`. This encoder is explicitly configured with `handle_unknown='use_encoded_value'` and `unknown_value=-1`. If production data contains a brand new state or business type, the model will not crash; it will handle it safely as an unknown class.

### 3. Multi-Model Probability & Sensitive Feature Export

We trained and evaluated three distinct algorithms (Logistic Regression, Random Forest with class-balancing, and HistGradientBoosting).

* Because `HistGradientBoosting` does not natively output feature importances, we integrated **Permutation Importance utilizing ROC AUC**. This evaluates subtle probability shifts on imbalanced data, saving a comprehensive $31$-column alignment table directly to `outputs/model_feature_importances.csv` alongside the model's predictions.



---

## Validation Results Summary

Our models were trained and validated across the three paradigms with the following results:

| Data Paradigm | Algorithm | Validation ROC-AUC | Validation Gini |
| --- | --- | --- | --- |
| **Sparse Data (NaNs Intact)** | **HistGradientBoosting** | **0.6687** | **0.3374** |
| Artificially Completed Data | RandomForest | 0.6679 | 0.3359 |
| Artificially Completed Data | HistGradientBoosting | 0.6642 | 0.3285 |
| Pruned Complete Data | RandomForest | 0.6587 | 0.3175 |
| Pruned Complete Data | HistGradientBoosting | 0.6320 | 0.2640 |
| Artificially Completed Data | LogisticRegression | 0.6071 | 0.2142 |
| Pruned Complete Data | LogisticRegression | 0.5756 | 0.1512 |

---

## 4. Production Readiness Roadmap

To move this model into an enterprise production environment, we recommend the following deployment roadmap:

### Phase 1: Code Modularization & Package Management

* **Object-Oriented Pipeline:** Transition the Jupyter notebook into structured Python modules (e.g., `src/data_preprocessing.py`, `src/train.py`, `src/predict.py`).
* **Environment Standardization:** Implement a containerized runtime environment using **Docker** and pin python package dependencies using a lockfile (e.g., `requirements.txt` or `poetry.lock`).

### Phase 2: Automated Testing & Validation Gates

* **Unit & Integration Tests:** Write tests (using `pytest`) to verify that the preprocessing engine properly handles missing data, capitalization formatting, and target-leak dropping.
* **Data Validation (Great Expectations):** Implement runtime assertions on input data schemas (e.g., verifying that categorical fields only contain expected structures, and numerical columns do not contain negative values).

### Phase 3: CI/CD & Model Registry (MLflow)

* **Continuous Integration / Continuous Deployment:** Set up GitHub Actions to run unit tests and automatically build a Docker image upon pushing to the main branch.
* **Model Versioning:** Integrate an ML registry tool (such as MLflow or W&B) to log hyperparameter metrics, artifacts, model weights, and the training data schema version.

### Phase 4: Serving Infrastructure & Drift Monitoring

* **API Serving Layer:** Wrap the inference engine in a RESTful API (using FastAPI) to serve predictions in real-time or via scheduled Airflow batch jobs.
* **Telemetry & Observability:** Monitor prediction distribution and input data over time (using libraries like `EvidentlyAI`) to trigger alert pipelines if feature drift occurs (e.g., if average driver age begins to shift significantly from the training baseline).
