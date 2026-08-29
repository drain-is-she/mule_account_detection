# Methodology

## 1. Problem Definition

The goal is to identify **potential money-mule accounts** from transaction
behavior and relationships within the transaction network.

The system is designed as a **defense-only risk detection and prioritization
tool**. A high-risk prediction indicates that an account resembles known mule
patterns; it is not treated as proof of fraud.

The pipeline combines two complementary sources of information:

1. **Behavioral transaction features**
2. **Graph-based relational features**

---

## 2. System Architecture

```text
Raw Transactions
       │
       ▼
Data Cleaning & Validation
       │
       ▼
Feature Engineering
       │
       ├───────────────┐
       ▼               ▼
 Behavioral        Transaction
  Features            Graph
       │               │
       ▼               ▼
   LightGBM         GraphSAGE
       │               │
       └───────┬───────┘
               ▼
        Risk Score Fusion
               │
               ▼
       Mule Risk Prediction
               │
               ▼
     Investigation Priority
