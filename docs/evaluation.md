## Evaluation

### Dataset & Validation Split

- **Total accounts in graph:** 179,562
- **Labeled accounts:** 47,874
  - Legitimate: 46,652 (97.45%)
  - Mules: 1,222 (2.55%)
- **Held-out validation set:** 9,575 accounts
  - Legitimate: 9,331
  - Mules: 244

> Primary performance metrics are reported on the held-out validation set.

### Confusion Matrix — Threshold 0.50

**Ensemble: 95% LightGBM + 5% GraphSAGE**

| | Predicted Legit | Predicted Mule |
|---|---:|---:|
| **Actual Legit** | **TN: 9,298** | **FP: 33** |
| **Actual Mule** | **FN: 88** | **TP: 156** |

| Metric | Score |
|---|---:|
| Accuracy | **98.74%** |
| ROC-AUC | **0.9113** |
| Precision | **82.54%** |
| Recall | **63.93%** |
| Specificity | **99.65%** |
| False Positive Rate | **0.35%** |
| F1-Score | **0.7206** |

**Interpretation:** The model catches **156/244 mule accounts (63.93%)** while incorrectly flagging only **33/9,331 legitimate accounts (0.35% FPR)**. The main remaining weakness is the **88 false negatives**.

### Threshold Analysis

The validation-set F1-optimal threshold is **0.1877**:

| | Threshold 0.50 | Threshold 0.1877 |
|---|---:|---:|
| False Positives | 33 | 37 |
| False Negatives | 88 | 83 |
| Precision | 82.54% | 81.31% |
| Recall | 63.93% | **65.98%** |
| F1 | 0.7206 | **0.7285** |

Lowering the threshold catches **5 additional mule accounts** at the cost of **4 additional false positives**.

### Model Comparison

| Metric | GraphSAGE | LightGBM | Ensemble |
|---|---:|---:|---:|
| ROC-AUC | 0.9085 | **0.9113** | **0.9113** |
| Precision | **84.87%** | 82.54% | 82.54% |
| Recall | 52.87% | **63.93%** | **63.93%** |
| F1 | 0.6515 | **0.7206** | **0.7206** |
| FP | **23** | 33 | 33 |
| FN | 115 | **88** | **88** |

**Key finding:** LightGBM carries most of the predictive signal. GraphSAGE provides complementary relational information, but its current aggregate lift is marginal. Further analysis focuses on identifying cases where graph information helps recover behavioral-model errors.

### Error & Cost Analysis

False positives represent unnecessary investigation of legitimate accounts; false negatives represent missed mule accounts.

Because the dataset does not provide real operational costs, monetary losses are **not claimed**. A normalized cost framework is used instead:

\[
Cost = FP \times C_{FP} + FN \times C_{FN}
\]

Detailed failure cases and error analysis are documented in [`failure_analysis.md`](failure_analysis.md).

> **Note:** The confusion matrix calculated across all 47,874 labeled accounts includes training observations and is therefore **not used as a generalization metric**.
