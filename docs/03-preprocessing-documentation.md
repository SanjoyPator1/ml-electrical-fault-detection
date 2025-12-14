# Phase 3: Data Preprocessing & Baseline Models

**Notebook:** `02_Preprocessing.ipynb`

**Status:** Completed

**Date Completed:** December 14, 2024

---

## Overview

Phase 3 focused on preparing the raw data for machine learning by implementing proper preprocessing techniques and establishing baseline model performance. This phase validates that our data pipeline is working correctly before moving to more advanced feature engineering and modeling.

---

## Objectives

1. Implement proper data preprocessing (scaling, encoding, splitting)
2. Create stratified train-test splits to maintain class distribution
3. Train baseline models to validate preprocessing effectiveness
4. Establish performance benchmarks for future comparison
5. Save all preprocessing artifacts for reproducibility

---

## Data Preparation

### Dataset Summary

- Total samples: 7,861
- Features: 6 (Ia, Ib, Ic, Va, Vb, Vc)
- Target classes: 6 fault types
- Class imbalance ratio: 2.36:1 (manageable)

### Fault Type Distribution

The dataset contains the following fault types:

```
No Fault: 2,365 samples (30.1%)
Line A Line B to Ground Fault: 1,134 samples (14.4%)
Three-Phase with Ground: 1,133 samples (14.4%)
Line-to-Line AB: 1,129 samples (14.4%)
Three-Phase: 1,096 samples (13.9%)
Line-to-Line with Ground BC: 1,004 samples (12.8%)
```

The imbalance is moderate and manageable without immediate need for SMOTE or other balancing techniques.

---

## Preprocessing Steps

### 1. Fault Type Label Creation

Combined the binary fault indicators (G, C, B, A) into a single categorical label representing the fault type. This simplifies the classification task from multi-label to multi-class.

**Mapping:**

- 0000 → No Fault
- 1011 → Line A Line B to Ground Fault
- 1111 → Three-Phase with Ground
- 1001 → Line-to-Line AB
- 0111 → Three-Phase
- 0110 → Line-to-Line with Ground BC

### 2. Feature Scaling

Applied MinMaxScaler to normalize all features to the range [0, 1].

**Rationale:**

- Current (Ia, Ib, Ic) and Voltage (Va, Vb, Vc) have different scales
- ML algorithms perform better with normalized features
- MinMaxScaler preserves the shape of distributions

**Result:**

- All features successfully scaled to [0.0, 1.0] range
- Distributions maintained their original shapes

### 3. Label Encoding

Converted fault type names to numerical labels for model training.

**Encoding:**

```
0: Line A Line B to Ground Fault
1: Line-to-Line AB
2: Line-to-Line with Ground BC
3: No Fault
4: Three-Phase
5: Three-Phase with Ground
```

### 4. Train-Test Split

Implemented stratified 80-20 split to maintain class distribution.

**Configuration:**

- Training set: 6,288 samples (80%)
- Test set: 1,573 samples (20%)
- Random state: 42 (for reproducibility)
- Stratified: Yes (maintains class proportions)

**Validation:**
Class distributions verified to be nearly identical across train and test sets, confirming successful stratification.

---

## Baseline Model Training

Three baseline models were trained to validate preprocessing and establish performance benchmarks.

### Models Tested

1. **Logistic Regression**

   - Linear baseline model
   - Configuration: max_iter=1000, random_state=42

2. **Decision Tree**

   - Non-linear baseline model
   - Configuration: random_state=42

3. **Random Forest**
   - Ensemble baseline model
   - Configuration: n_estimators=100, random_state=42

### Cross-Validation Strategy

- Method: 5-fold Stratified K-Fold
- Ensures each fold maintains class distribution
- Provides reliable performance estimates

---

## Results

### Model Performance Summary

```
Model                    CV Accuracy    Test Accuracy
-------------------------------------------------------
Logistic Regression      33.81%         34.58%
Decision Tree            86.31%         88.56%
Random Forest            85.99%         87.98%
```

### Key Observations

1. **Decision Tree** performed best with 88.56% test accuracy

   - Good generalization (CV and test scores are close)
   - Can capture non-linear patterns in fault data
   - Shows minimal overfitting

2. **Random Forest** performed similarly at 87.98%

   - Slightly lower than Decision Tree at this stage
   - Ensemble benefits may become more apparent with feature engineering
   - More stable across different data splits

3. **Logistic Regression** struggled at 34.58%
   - Limited by linear decision boundaries
   - Electrical fault patterns are inherently non-linear
   - Expected to improve significantly with polynomial features

### Confusion Matrix Analysis

**Best Model (Decision Tree) Performance by Class:**

```
Class                              Precision  Recall   F1-Score  Support
------------------------------------------------------------------------
Line A Line B to Ground Fault      1.00       0.99     0.99      227
Line-to-Line AB                    0.99       1.00     0.99      226
Line-to-Line with Ground BC        1.00       1.00     1.00      201
No Fault                           1.00       1.00     1.00      473
Three-Phase                        0.60       0.61     0.60      219
Three-Phase with Ground            0.61       0.60     0.61      227
```

**Major Finding:**
The model struggles to distinguish between "Three-Phase" and "Three-Phase with Ground" faults, with 175 total misclassifications between these two classes. This is an expected challenge that will be addressed with domain-specific feature engineering in Phase 4.

**Why This Confusion Occurs:**

- Both fault types involve all three phases
- Ground connection creates subtle electrical differences
- Similar current and voltage magnitudes
- Requires specialized features (zero-sequence components) to distinguish

---

## Key Insights

### What Worked Well

1. **Preprocessing Pipeline**

   - Feature scaling successfully normalized all features
   - Stratified split maintained class balance
   - No data leakage detected

2. **Tree-Based Models**

   - Decision Tree and Random Forest perform well (~88%)
   - Can handle non-linear fault patterns
   - Good starting point for further optimization

3. **Data Quality**
   - No missing values
   - No duplicates
   - Clean, well-structured dataset

### Identified Challenges

1. **Linear Models Struggle**

   - Logistic Regression only achieves 34.58%
   - Fault classification requires non-linear decision boundaries
   - Need feature engineering to help linear models

2. **Three-Phase Fault Confusion**

   - Three-Phase vs Three-Phase with Ground are difficult to distinguish
   - 175 misclassifications between these classes
   - Require domain-specific features (zero-sequence components)

3. **Limited Feature Set**
   - Currently using only 6 raw measurements
   - No interaction terms or derived features
   - Large room for improvement through feature engineering

---

## Files Generated

### Preprocessed Data

All saved in `data/processed/`:

- `X_train.npy` - Training features (6,288 samples)
- `X_test.npy` - Test features (1,573 samples)
- `y_train.npy` - Training labels
- `y_test.npy` - Test labels
- `feature_names.txt` - List of feature names

### Models and Preprocessing Artifacts

All saved in `models/`:

- `minmax_scaler.pkl` - Fitted MinMaxScaler
- `label_encoder.pkl` - Fitted LabelEncoder
- `baseline_logistic_regression.pkl` - Trained Logistic Regression
- `baseline_decision_tree.pkl` - Trained Decision Tree
- `baseline_random_forest.pkl` - Trained Random Forest

### Visualizations

All saved in `results/preprocessing/plots/`:

1. `01_fault_type_distribution.png` - Bar chart of fault type counts
2. `02_features_before_scaling.png` - Feature distributions before scaling
3. `03_features_after_scaling.png` - Feature distributions after scaling
4. `04_train_test_split_distribution.png` - Class distribution in train/test sets
5. `05_baseline_model_comparison.png` - Performance comparison of 3 models
6. `06_confusion_matrix_best_baseline.png` - Detailed confusion matrix

### Metrics

All saved in `results/preprocessing/metrics/`:

- `baseline_model_results.csv` - Summary of all model performances
- `best_baseline_classification_report.csv` - Detailed metrics for best model
- `confusion_matrix_best_baseline.csv` - Raw confusion matrix data

### Summary

- `PREPROCESSING_SUMMARY.txt` - Complete text summary of Phase 3

---

## Validation Checklist

All preprocessing steps were validated:

```
PASS: Data scaling - All features in [0, 1] range
PASS: Stratified split - Class distributions match
PASS: No data leakage - Train/test sets are disjoint
PASS: Models saved - All 3 baseline models persisted
PASS: Preprocessors saved - Scaler and encoder saved
PASS: All models trained - 3 models successfully trained
```

---

## Expected Improvements for Phase 4

Based on baseline results, we anticipate the following improvements from feature engineering:

### From Polynomial Features (degree 2)

- Creates interaction terms (e.g., Ia*Ib, Va*Vb)
- Expected improvement: +8-12% for all models
- Particularly beneficial for Logistic Regression
- Decision Tree: 88% → 95%+
- Random Forest: 88% → 95%+
- Logistic Regression: 35% → 60%+

### From Domain-Based Features

- Zero-sequence current: (Ia + Ib + Ic) / 3
- Zero-sequence voltage: (Va + Vb + Vc) / 3
- Phase angle differences
- THD approximations
- Voltage/Current ratios
- Expected improvement: +3-5% for tree models
- Should help distinguish Three-Phase faults

---

## Next Steps

**Phase 4: Feature Engineering**

Priority tasks:

1. Create polynomial features (degree 2) from existing features
2. Implement domain-based electrical features:
   - Zero-sequence components
   - Phase angle differences
   - Total Harmonic Distortion (THD) approximations
   - Voltage/Current ratios
3. Retrain baseline models with new features
4. Compare performance improvements
5. Identify most important features

**Success Criteria for Phase 4:**

- Decision Tree accuracy > 95%
- Random Forest accuracy > 95%
- Logistic Regression accuracy > 60%
- Reduced confusion between Three-Phase faults

---

## Technical Notes

### Reproducibility

- Random seed set to 42 throughout
- All preprocessing artifacts saved
- Exact train-test indices preserved in splits

### Computational Efficiency

- Preprocessing completed in < 5 seconds
- Model training completed in < 30 seconds
- Total notebook runtime < 2 minutes

### Memory Usage

- Raw data: ~0.6 MB
- Processed data: ~1.2 MB
- Saved models: ~0.8 MB
- Total storage: ~2.6 MB

---

## Lessons Learned

1. **Always validate preprocessing**: The validation checklist caught several potential issues early
2. **Stratification is crucial**: Class imbalance would have been worse without stratified splitting
3. **Tree models are natural fit**: Decision trees handle electrical fault patterns well
4. **Feature engineering is needed**: Linear models and Three-Phase confusion indicate need for better features
5. **Start simple**: Baseline models provide valuable insights before adding complexity

---

## Conclusion

Phase 3 successfully established a solid preprocessing pipeline and baseline model performance. The Decision Tree achieves 88.56% accuracy with just 6 basic features, validating our approach. The identified challenges (linear model performance, Three-Phase confusion) provide clear direction for Phase 4 feature engineering.

All preprocessing artifacts are saved and ready for use in subsequent phases. The project is on track, with a clear roadmap for achieving target performance (>95% accuracy) through systematic feature engineering and model optimization.

**Status:** Ready to proceed to Phase 4 - Feature Engineering
