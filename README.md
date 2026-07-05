# Propositional Logic and Naive Bayes

## 🧭 Project Overview

### Title:
Propositional Logic and Naive Bayes

### Purpose:

This repository contains two Artificial Intelligence projects focused on symbolic reasoning and probabilistic machine learning. The first implements propositional logic entailment using PL-Resolution and DPLL algorithms, while the second develops a Multinomial Naive Bayes classifier for text classification using multiple preprocessing techniques and performance evaluation.

### Audience:

- Artificial Intelligence Students
- Computer Science Students
- Machine Learning Enthusiasts
- Researchers
- Software Engineers

---

# 🧱 Project Scope

## Task 1: Propositional Logic

Implements algorithms for determining logical entailment from a propositional knowledge base.

### Components

- Knowledge Base Parser
- Clause Representation
- PL-Resolution Engine
- DPLL SAT Solver
- Runtime Benchmarking
- Performance Visualization

### Techniques

- Clause Resolution
- Resolution Refutation
- Davis–Putnam–Logemann–Loveland (DPLL)
- Unit Clause Propagation
- Pure Symbol Heuristic
- Runtime Scalability Analysis

---

## Task 2: Naive Bayes Text Classification

Implements a Multinomial Naive Bayes classifier capable of classifying text documents across multiple categories while evaluating different preprocessing pipelines.

### Components

- Text Preprocessing Pipeline
- Training Dataset Loader
- Testing Dataset Loader
- Multinomial Naive Bayes Classifier
- Evaluation Module
- Visualization Module

### Techniques

- Multinomial Naive Bayes
- Laplace Smoothing
- Log Probability Computation
- Confusion Matrix Generation
- Precision, Recall, and F1 Evaluation
- Misclassification Analysis

---

# 📂 Repository Structure

```text
Project3/
│
├── README.md  
├── Task1_Propositional_Logic/  
│   ├── PL_Titagwan_Bradley.ipynb  
│   └── report/  
│       └── Propositional_Logic_Report.pdf  
│  
├── Task2_Naive_Bayes/  
│   ├── NB_Titagwan_Bradley.py  
│   ├── data/  
│   │   ├── training_biology.txt  
│   │   ├── training_economics.txt  
│   │   ├── training_history.txt  
│   │   ├── training_physics.txt  
│   │   ├── testing_data.txt  
│   │   └── train.csv  
│   ├── results (screenshots)/  
│   │   ├── extra_accuracy_comparison  
│   │   ├── extra_confusion_lower_only  
│   │   ├── extra_confusion_lower_punct  
│   │   ├── extra_confusion_lower_punct_stop  
│   │   ├── extra_confusion_lower_punct_stop_alpha05  
│   │   |── extra_confusion_lower_punct_stop_alpha15  
│   │   ├── provided_accuracy_comparison  
│   │   ├── provided_confusion_lower_only  
│   │   ├── provided_confusion_lower_punct  
│   │   ├── provided_confusion_lower_punct_stop  
│   │   ├── provided_confusion_lower_punct_stop_alpha05  
│   │   └── provided_confusion_lower_punct_stop_alpha15  
│   └── report/  
        └── Naive_Bayes_Report.pdf  

```

---

# 🤖 Algorithms Included

| Task | Algorithm | Purpose |
|------|-----------|----------|
| Task 1 | PL-Resolution | Logical entailment through clause resolution |
| Task 1 | DPLL | Efficient satisfiability checking using recursive search |
| Task 2 | Multinomial Naive Bayes | Probabilistic text classification |
| Task 2 | Laplace Smoothing | Prevent zero-probability predictions |
| Task 2 | Text Preprocessing | Lowercasing, punctuation removal, stopword removal, lemmatization |

---

# 📊 Results Summary

## Propositional Logic

The benchmarking experiments demonstrated that:

- Both PL-Resolution and DPLL correctly determined logical entailment.
- DPLL consistently outperformed PL-Resolution as the number of variables and clauses increased.
- Runtime comparison plots illustrate the scalability advantages of DPLL for larger knowledge bases.

## Naive Bayes

The classifier was evaluated using multiple preprocessing configurations on two datasets.

### Provided Dataset

- Best Accuracy: **93%**
- Best Configuration:
  - Lowercase conversion
  - Punctuation removal
  - Stopword removal
  - Laplace smoothing

### External Dataset

- Evaluated using the first **1,200** samples for training and the remaining samples for testing.
- Multiple preprocessing pipelines were compared using confusion matrices and accuracy plots.
- The external dataset produced competitive performance while highlighting the effects of different preprocessing strategies.

---

# 🚀 How to Run

## Task 1 — Propositional Logic

Launch the notebook:

```bash
jupyter notebook PL_Titagwan_Bradley_updated.ipynb
```

The notebook will:

- Test propositional logic entailment
- Compare PL-Resolution and DPLL
- Generate runtime comparison graphs
- Measure algorithm scalability

---

## Task 2 — Naive Bayes

Run the classifier:

```bash
python NB_Titagwan_Bradley.py
```

The program will:

- Train a Multinomial Naive Bayes classifier
- Evaluate multiple preprocessing configurations
- Generate confusion matrices
- Generate accuracy comparison plots
- Compute Precision, Recall, and F1-score
- Display misclassification analysis

---

# 💻 Skills Demonstrated

- Artificial Intelligence
- Symbolic AI
- Knowledge Representation
- Automated Reasoning
- SAT Solving
- Machine Learning
- Natural Language Processing
- Text Classification
- Data Preprocessing
- Performance Benchmarking
- Algorithm Analysis
- Python
- NumPy
- Matplotlib
- NLTK

---

# 📈 Future Improvements

- Implement additional SAT solving heuristics
- Expand benchmarking with larger knowledge bases
- Evaluate additional machine learning classifiers
- Incorporate TF-IDF feature extraction
- Compare Naive Bayes against Logistic Regression and Support Vector Machines

---

# 📬 Author

**Bradley Titagwan**

Version: v1.0


