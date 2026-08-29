1. Dataset Partition Summary
Total Accounts in Graph: 179,562
Labeled Accounts: 47,874 (Legitimate: 46,652 [97.45%], Mules: 1,222 [2.55%])
Validation Split (20% Holdout): 9,575 accounts
Actual Legitimate (Class 0): 9,331 (97.45%)
Actual Mules (Class 1): 244 (2.55%)
2. Validation Set Confusion Matrix (Primary Metric)
A. Optimal Ensemble (
5
%
5% GraphSAGE + 
95
%
95% LightGBM) @ Decision Threshold = 
0.50
0.50
text


                             PREDICTED CLASS
                     ┌──────────────────┬──────────────────┐
                     │ Predicted Legit  │  Predicted Mule  │
  ┌──────────────────┼──────────────────┼──────────────────┤
A │ Actual Legit (0) │  TN = 9,298      │   FP = 33        │  9,331 Total
C ├──────────────────┼──────────────────┼──────────────────┤
T │ Actual Mule (1)  │  FN = 88         │   TP = 156       │    244 Total
U └──────────────────┴──────────────────┴──────────────────┘
A                      9,386 Total          189 Total
L
Metric	Value	Technical Formula	AML Operational Meaning
Accuracy	98.74%	
T
P
+
T
N
T
o
t
a
l
Total
TP+TN
​
 	Overall correct classification rate
ROC-AUC	0.9113	Area Under Curve	Model ranking quality across all thresholds
Precision (PPV)	82.54%	
T
P
T
P
+
F
P
=
156
189
TP+FP
TP
​
 = 
189
156
​
 	Of all accounts flagged as mules, 82.5% are true mules (17.5% false alarms).
Recall / Sensitivity	63.93%	
T
P
T
P
+
F
N
=
156
244
TP+FN
TP
​
 = 
244
156
​
 	Caught 156 out of 244 active mules (missed 88 mules).
Specificity (TNR)	99.65%	
T
N
T
N
+
F
P
=
9298
9331
TN+FP
TN
​
 = 
9331
9298
​
 	Correctly cleared 99.65% of legitimate accounts.
False Positive Rate	0.35%	
F
P
T
N
+
F
P
=
33
9331
TN+FP
FP
​
 = 
9331
33
​
 	Low friction for legitimate banking customers.
F1-Score	0.7206	
2
⋅
P
⋅
R
P
+
R
2⋅ 
P+R
P⋅R
​
 	Harmonic balance between Precision and Recall
B. Optimal Ensemble @ Best F1 Threshold = 
0.1877
0.1877
When optimizing threshold for maximum F1-Score:

text


                             PREDICTED CLASS
                     ┌──────────────────┬──────────────────┐
                     │ Predicted Legit  │  Predicted Mule  │
  ┌──────────────────┼──────────────────┼──────────────────┤
A │ Actual Legit (0) │  TN = 9,294      │   FP = 37        │  9,331 Total
C ├──────────────────┼──────────────────┼──────────────────┤
T │ Actual Mule (1)  │  FN = 83         │   TP = 161       │    244 Total
U └──────────────────┴──────────────────┴──────────────────┘
Precision: 81.31%
Recall: 65.98% (+5 more mules caught)
F1-Score: 0.7285 (Peak F1)
3. Model Comparison: Single Modalities vs. Ensemble
Metric	GraphSAGE (GNN Only)	LightGBM Only	Optimal Ensemble
True Negatives (TN)	9,308	9,298	9,298
False Positives (FP)	23	33	33
False Negatives (FN)	115	88	88
True Positives (TP)	129	156	156
ROC-AUC	0.9085	0.9113	0.9113
Precision	84.87%	82.54%	82.54%
Recall	52.87%	63.93%	63.93%
F1-Score	0.6515	0.7206	0.7206
4. Confusion Matrix across All Labeled Data (
47
,
874
47,874 Accounts)
Combining Training (
38
,
299
38,299) and Validation (
9
,
575
9,575) sets at threshold 
0.50
0.50:

text


                             PREDICTED CLASS
                     ┌──────────────────┬──────────────────┐
                     │ Predicted Legit  │  Predicted Mule  │
  ┌──────────────────┼──────────────────┼──────────────────┤
A │ Actual Legit (0) │  TN = 46,619     │   FP = 33        │  46,652 Total
C ├──────────────────┼──────────────────┼──────────────────┤
T │ Actual Mule (1)  │  FN = 88         │   TP = 1,134     │   1,222 Total
U └──────────────────┴──────────────────┴──────────────────┘
Overall Precision: 97.17%
Overall Recall: 92.80%
Overall F1-Score: 0.9494
