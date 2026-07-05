# Propositional Logic and Naive Bayes

### PL-Resolution, DPLL, and Multinomial Naive Bayes Text Classification

Author: Bradley Titagwan

This repository contains my full implementation for Project 3, covering:

- Propositional Logic Entailment
- PL-Resolution
- DPLL Satisfiability Checking
- Runtime comparison of PL-Resolution vs DPLL
- Multinomial Naive Bayes text classification
- Preprocessing experiments and smoothing comparison

---

## Repository Structure

Project3/

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
│       └── Naive_Bayes_Report.pdf  
│  
└── LICENSE  

---

## How to Run

### Propositional Logic

Open the notebook:

```bash
jupyter notebook PL_Titagwan_Bradley_updated.ipynb
