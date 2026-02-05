# Phase 6: K-Nearest Neighbors Model

**Notebook:** `05_KNN_Model.ipynb`

**Status:** Completed

**Date Completed:** February 5, 2026

---

## Overview

Phase 6 implemented and evaluated K-Nearest Neighbors (KNN) as an alternative classification approach for electrical fault detection. While Phase 5 achieved 99.75% accuracy with LightGBM, this phase explores KNN to provide a comprehensive comparison with instance-based learning methods and validate that gradient boosting algorithms are indeed the optimal choice for this problem.

---

## Objectives

1. Implement baseline KNN classifier with default parameters
2. Perform comprehensive hyperparameter tuning using grid search
3. Evaluate KNN performance against Phase 4 and Phase 5 models
4. Analyze computational efficiency (training and inference time)
5. Provide recommendations on KNN suitability for fault detection

---

## K-Nearest Neighbors Theory

### Algorithm Overview

K-Nearest Neighbors is a non-parametric, instance-based learning algorithm that classifies samples based on proximity to training examples. Unlike model-based algorithms that learn explicit decision boundaries, KNN stores the entire training dataset and makes predictions by finding the k closest training samples to a query point.

**Classification Process:**

1. Store all training samples in memory
2. For a new sample, calculate distance to all training samples
3. Identify k nearest neighbors
4. Assign class by majority vote among k neighbors
5. Optionally weight votes by inverse distance

### Distance Metrics

**Euclidean Distance (default):**

```
d(x, y) = √(Σ(xi - yi)²)
```

**Manhattan Distance:**

```
d(x, y) = Σ|xi - yi|
```

**Minkowski Distance (generalization):**

```
d(x, y) = (Σ|xi - yi|^p)^(1/p)
```

Where p=1 gives Manhattan, p=2 gives Euclidean

### Key Hyperparameters

**n_neighbors (k):**

The number of nearest neighbors to consider for voting.

- Small k (3-5): More sensitive to noise, complex decision boundaries, prone to overfitting
- Large k (11-15): Smoother decision boundaries, more robust to noise, may underfit
- Typical values: 3, 5, 7, 9, 11

**weights:**

How to weight the contribution of neighbors.

- `uniform`: All k neighbors have equal vote
- `distance`: Closer neighbors have more influence (weight = 1/distance)

**metric:**

The distance function used to find neighbors.

- `euclidean`: Standard straight-line distance
- `manhattan`: Sum of absolute differences
- `minkowski`: Generalized distance metric

### Advantages

1. **Simple and intuitive** - Easy to understand and explain
2. **No training phase** - Model is the data itself
3. **Naturally handles multi-class** - No modifications needed
4. **Non-parametric** - Makes no assumptions about data distribution
5. **Adaptive decision boundaries** - Can model complex patterns

### Disadvantages

1. **Slow prediction** - Must compare to all training samples
2. **Memory intensive** - Stores entire training dataset
3. **Curse of dimensionality** - Performance degrades with many features
4. **Sensitive to feature scaling** - Requires normalized features
5. **Poor with imbalanced classes** - Dominated by majority class
6. **No model interpretability** - Cannot extract feature importance

---

## Implementation

### Dataset Configuration

Using the best-performing features from Phase 4:

- Training samples: 6,288
- Test samples: 1,573
- Number of features: 29 (domain-engineered features)
- Number of classes: 6 fault types
- Feature scaling: MinMaxScaler (applied in Phase 3)

**Fault Types:**

- ABC (Line A Line B to Ground Fault)
- ABCG (Line-to-Line AB)
- ABG (Line-to-Line with Ground BC)
- AG (No Fault)
- BC (Three-Phase)
- Normal (Three-Phase with Ground)

---

## Baseline KNN Model

### Configuration

Default KNN parameters to establish baseline performance:

```python
n_neighbors = 5
weights = 'uniform'
metric = 'euclidean'
n_jobs = -1  # Use all CPU cores
```

### Cross-Validation Results

5-fold stratified cross-validation:

```
CV Accuracy: 77.27% (±0.35%)
CV Time: 1.94 seconds
```

**Analysis:**

Moderate baseline performance with low variance indicates stable predictions but insufficient accuracy for fault detection. The 77% accuracy is significantly lower than Phase 4 Random Forest (99.43%), suggesting KNN struggles with the complexity of this problem.

### Test Set Performance

```
Test Accuracy: 75.46%
Precision: 70.19%
Recall: 71.23%
F1-Score: 70.67%

Training Time: 0.0047 seconds
Prediction Time: 0.0601 seconds
Average per sample: 0.038 ms
```

**Key Observations:**

1. **Underfitting detected** - 75% accuracy is inadequate for fault detection
2. **CV-Test discrepancy** - Test accuracy lower than CV suggests some instability
3. **Fast training** - Instant training as KNN is lazy learning
4. **Slow inference** - 60ms for 1,573 samples is concerning for real-time systems

### Per-Class Performance (Baseline)

From confusion matrix analysis:

**Strong Performance:**

- ABG: Perfect classification (100% precision and recall)
- AG (No Fault): Perfect classification (100% on 473 samples)
- ABCG: Good performance (93.8% precision)

**Weak Performance:**

- BC: Only 29.7% correctly classified (65/219)
- Normal: Only 23.3% correctly classified (53/227)
- ABC: 81.1% correctly classified but 27 misclassifications

**Major Confusion:**

- BC misclassified as Normal: 133 samples (60.7%)
- Normal misclassified as BC: 141 samples (62.1%)

This bidirectional confusion indicates that BC and Normal fault types have very similar patterns in the feature space, making them nearly indistinguishable to KNN.

---

## Hyperparameter Tuning

### Grid Search Configuration

Comprehensive search over key hyperparameters:

```python
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

Total combinations: 36
CV Strategy: 5-fold Stratified K-Fold
```

### Grid Search Results

```
Total Time: 5.80 seconds (0.10 minutes)
Best CV Accuracy: 85.70%

Best Parameters:
- n_neighbors: 3
- weights: distance
- metric: manhattan
```

**Analysis:**

1. **Smaller k optimal** - k=3 outperforms larger values, suggesting local patterns matter
2. **Distance weighting helps** - Closer neighbors provide more reliable votes
3. **Manhattan distance better** - Manhattan outperforms Euclidean for this feature space
4. **Significant improvement** - 8.43% gain over baseline (77.27% to 85.70%)

### Top Parameter Combinations

The best 5 configurations:

```
Rank 1: 85.70% (k=3, weights=distance, metric=manhattan)
Rank 2: 85.54% (k=3, weights=distance, metric=minkowski)
Rank 3: 85.45% (k=3, weights=distance, metric=euclidean)
Rank 4: 84.71% (k=5, weights=distance, metric=manhattan)
Rank 5: 84.63% (k=5, weights=distance, metric=minkowski)
```

**Key Findings:**

- Distance weighting appears in all top 5 configurations
- k=3 consistently outperforms larger k values
- Metric choice has smaller impact than k and weighting

---

## Final Model Performance

### Tuned KNN Test Results

```
Test Accuracy: 89.32%
Precision: 87.41%
Recall: 87.49%
F1-Score: 87.45%

Prediction Time: 0.0507 seconds
Average per sample: 0.032 ms
```

**Improvement Over Baseline:**

- Accuracy: +13.86% (75.46% → 89.32%)
- Precision: +17.22% (70.19% → 87.41%)
- Recall: +16.26% (71.23% → 87.49%)
- F1-Score: +16.78% (70.67% → 87.45%)

### Per-Class Performance (Tuned)

```
Class    Precision  Recall   F1-Score  Support
-----    ---------  ------   --------  -------
ABC      97.76%     96.04%   96.89%    227
ABCG     97.00%     100.00%  98.47%    226
ABG      100.00%    100.00%  100.00%   201
AG       100.00%    99.79%   99.89%    473
BC       64.95%     63.47%   64.20%    219
Normal   64.78%     65.64%   65.21%    227
```

**Analysis:**

**Excellent Performance (>95% F1):**

- ABG: Perfect classification maintained
- AG (No Fault): Near-perfect, only 1 error
- ABCG: Perfect recall, minimal false positives
- ABC: Excellent across all metrics

**Poor Performance (<70% F1):**

- BC (Three-Phase): 64.20% F1-Score
- Normal: 65.21% F1-Score

The BC vs Normal confusion remains the primary weakness, though improved from baseline. These two classes are inherently difficult to distinguish in the feature space.

### Confusion Matrix Analysis (Tuned)

**Major Misclassifications:**

```
BC predicted as Normal: 79 samples (36.1%)
Normal predicted as BC: 74 samples (32.6%)
```

Compared to baseline (60.7% and 62.1%), this is significant improvement but still problematic. The tuned model reduced misclassification by 40% but did not eliminate the fundamental confusion.

**Other Notable Errors:**

- ABC → ABCG: 7 samples (minor confusion)
- ABC → BC: 1 sample (negligible)
- ABC → Normal: 1 sample (negligible)
- AG → Normal: 1 sample (negligible)
- ABG → Normal: 1 sample (negligible)

Most errors are concentrated in the BC-Normal boundary, which appears to be a fundamental limitation of the feature space rather than the algorithm.

---

## Comparison with Previous Phases

### Model Performance Ranking

```
Rank  Model                    Test Accuracy  CV Accuracy
----  -----                    -------------  -----------
1     LightGBM (Phase 5)       99.75%         99.65%
2     XGBoost (Phase 5)        99.62%         99.44%
3     Random Forest (Phase 4)  99.43%         99.32%
4     KNN (Tuned)              89.32%         85.70%
5     KNN (Baseline)           75.46%         77.27%
```

**Accuracy Gap:**

- LightGBM vs KNN Tuned: 10.43% (99.75% - 89.32%)
- Random Forest vs KNN Tuned: 10.11% (99.43% - 89.32%)

This substantial gap demonstrates that gradient boosting and ensemble methods significantly outperform instance-based learning for this problem.

### Speed Comparison

```
Model                  Prediction Time (s)  Relative Speed
-----                  -------------------  --------------
LightGBM              0.05                 1.0x (fastest)
KNN Tuned             0.051                1.02x
XGBoost               0.06                 1.2x
Random Forest         0.08                 1.6x
KNN Baseline          0.060                1.2x
```

**Analysis:**

Surprisingly, KNN prediction time is competitive with gradient boosting methods for this dataset size. However, this advantage disappears at scale:

- KNN time grows linearly with training set size
- Tree-based methods maintain constant prediction time
- With 10x more data, KNN would be 10x slower while LightGBM remains fast

### Training Time Comparison

```
Model                  Training Time (s)
-----                  -----------------
KNN (any config)       ~0.005 (no training)
XGBoost                0.84
KNN Grid Search        5.80 (for tuning)
LightGBM               1.98
Random Forest          2.50
```

KNN has no training phase (lazy learning), but hyperparameter tuning via grid search negates this advantage.

---

## Performance Analysis

### Why KNN Underperforms

**1. Curse of Dimensionality**

With 29 features, the feature space is large. In high dimensions:

- Distance metrics become less meaningful
- All points appear roughly equidistant
- Nearest neighbors may not be truly similar

**2. Complex Decision Boundaries**

Electrical faults create non-linear, complex patterns that require:

- Feature interactions (product terms)
- Hierarchical decision logic
- Adaptive boundary complexity

KNN uses simple distance metrics that cannot capture these patterns as effectively as tree-based methods.

**3. Class Imbalance Effects**

KNN is sensitive to class imbalance. With "No Fault" having 30% of samples:

- Local neighborhoods may be dominated by majority class
- Distance weighting helps but doesn't fully compensate
- Tree-based methods handle imbalance more robustly

**4. Feature Scale Sensitivity**

While we applied MinMaxScaler, the features still vary in importance. KNN treats all features equally in distance calculations, whereas tree-based methods learn feature importance.

### Where KNN Succeeds

**Clear Fault Types:**

KNN achieved 100% accuracy on ABG (Line-to-Line with Ground BC) because these faults have:

- Distinct signature in feature space
- Well-separated clusters
- Consistent patterns within class

**Stable Predictions:**

Low CV standard deviation (±0.35%) indicates KNN produces stable predictions across different data splits.

### Where KNN Fails

**Ambiguous Fault Types:**

BC and Normal classes have:

- Overlapping feature distributions
- Subtle distinctions requiring complex logic
- Non-linear decision boundaries

Tree-based methods handle this via hierarchical splits, while KNN relies on simple distance.

---

## Computational Analysis

### Memory Requirements

```
Training Data Storage: 6,288 samples × 29 features × 8 bytes = ~1.4 MB
Model Size: ~1.4 MB (stores entire training set)

Comparison:
- LightGBM model: ~50 KB
- XGBoost model: ~100 KB
- Random Forest model: ~500 KB
```

KNN is 10-30x larger than gradient boosting models.

### Scalability Analysis

**Current Dataset:**

- Prediction time: 51ms for 1,573 samples
- Per-sample latency: 32 microseconds

**Projected for 10x Data (62,880 training samples):**

- Prediction time: ~510ms (10x slower)
- Per-sample latency: 324 microseconds

**Projected for 100x Data (628,800 training samples):**

- Prediction time: ~5.1s (100x slower)
- Per-sample latency: 3,240 microseconds

Tree-based methods maintain constant O(log n) prediction time regardless of training set size, while KNN grows linearly O(n).

### Real-Time Deployment Considerations

For real-time fault detection systems:

**Requirements:**

- Response time: <10ms per fault event
- Throughput: 100+ predictions per second
- Low memory footprint for edge devices

**KNN Assessment:**

- Current 51ms for batch prediction is acceptable
- But 32μs per sample would become 320μs with 10x more data
- Memory requirements grow with every new data point
- Not suitable for online learning scenarios

**Recommendation:**

KNN is unsuitable for production fault detection systems due to poor scalability and memory constraints.

---

## Key Findings

### Quantitative Results

1. **Tuned KNN achieved 89.32% accuracy** - respectable but insufficient for fault detection
2. **10.43% accuracy gap** - substantial difference from LightGBM (99.75%)
3. **Distance weighting essential** - improved performance by 8.4%
4. **k=3 optimal** - smaller neighborhoods perform better
5. **Manhattan distance superior** - outperforms Euclidean for this problem

### Qualitative Insights

1. **BC-Normal confusion persistent** - fundamental limitation in feature space
2. **Clear faults classified perfectly** - ABG, AG achieve 99-100% accuracy
3. **Scalability concerns** - linear growth in prediction time
4. **Memory intensive** - stores entire training set
5. **No interpretability** - cannot extract feature importance or rules

### Comparison Insights

1. **Gradient boosting dominates** - 10% higher accuracy than KNN
2. **Ensemble methods excel** - tree-based approaches handle complexity better
3. **KNN competitive on speed** - for current dataset size only
4. **Training time misleading** - KNN has no training but requires tuning

---

## Recommendations

### For Production Deployment

**Do NOT use KNN for:**

- Real-time electrical fault detection systems
- Applications requiring scalability
- Systems with limited memory
- Scenarios needing model interpretability

**Use LightGBM or XGBoost instead:**

- 10% higher accuracy (99.75% vs 89.32%)
- Constant prediction time regardless of data size
- 10-30x smaller model size
- Feature importance available
- Proven scalability

### Appropriate Use Cases for KNN

**Use KNN for:**

1. **Small-scale applications** - Less than 1,000 training samples
2. **Research and education** - Understanding instance-based learning
3. **Baseline comparisons** - Establishing lower performance bounds
4. **Prototyping** - Quick initial experiments before advanced methods
5. **Interpretable predictions** - When understanding individual cases matters

### Improvements for KNN (Not Recommended)

If KNN must be used despite limitations:

1. **Dimensionality reduction** - PCA or feature selection to reduce curse
2. **Approximate nearest neighbors** - Libraries like FAISS for speed
3. **Local outlier detection** - Remove noisy samples from training set
4. **Class weighting** - Adjust for imbalance explicitly
5. **Ensemble KNN** - Multiple k values with voting

However, these improvements are unlikely to close the 10% accuracy gap with gradient boosting.

---

## Conclusion

Phase 6 successfully evaluated K-Nearest Neighbors for electrical fault detection and conclusively demonstrated its inferiority to gradient boosting methods.

**Summary:**

- **KNN Tuned Accuracy:** 89.32%
- **LightGBM Accuracy:** 99.75%
- **Performance Gap:** 10.43%
- **Speed:** Competitive initially but scales poorly
- **Memory:** 10-30x larger than tree-based models

**Key Takeaway:**

While KNN achieved respectable performance and proved useful for validating our feature engineering approach, it is unsuitable for production electrical fault detection systems. The combination of lower accuracy, poor scalability, and memory constraints make gradient boosting the clear choice.

**Final Recommendation:**

Continue using **LightGBM from Phase 5** as the primary model for electrical fault detection. KNN serves as a valuable baseline for comparison and educational purposes but should not be deployed in operational systems.

**Thesis Contribution:**

This comprehensive evaluation strengthens the thesis by:

1. Demonstrating systematic comparison of multiple algorithm families
2. Validating that performance differences are not due to feature engineering
3. Providing quantitative evidence for gradient boosting superiority
4. Establishing clear deployment recommendations backed by data

---

**End of Documentation**

This document provides comprehensive understanding of Phase 6 KNN Model, covering theory, implementation, detailed results analysis, and practical deployment considerations.
