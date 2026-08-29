# Failure Analysis

## Purpose

This evaluates the confusion matrix which depicts where the model had actually failed 

## False Positives

A false positive occurs when a legitimate account is classified
as a mule.

### Summary

- Total legitimate accounts: 9,331
- False positives: 33
- False-positive rate: 0.35%

### Exception 

**Account:** XXXXX

- Actual class: Legitimate
- Predicted class: Mule
- Risk score: 0.82

#### Why was it flagged?

- High transaction velocity
- High number of counterparties
- Unusual transaction concentration
- Network structure resembling mule behavior

#### Why was this likely a false positive?

The account exhibits legitimate high-volume behavior that overlaps
with patterns learned by the model as suspiciou

---

## False Negatives

A false negative occurs when a mule account is classified
as legitimate.

### Summary

- Actual mule accounts: 244
- False negatives: 88
- False-negative rate: 36.07%

### Exception 

**Account:** XXXXX

- Actual class: Mule
- Predicted class: Legitimate
- Risk score: 0.31

#### Why was it missed?

- Low transaction volume
- Limited graph connectivity
- Weak behavioral anomaly
- Insufficient network evidence

#### What could improve detection?

Investigate whether graph/community features provide additional
signal for this class of low-activity mule accounts.
