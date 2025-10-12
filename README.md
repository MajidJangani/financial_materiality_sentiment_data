## ** ESG Materiality Analysis: Sentiment-Driven Financial Forecasting**


```markdown
# ESG Materiality Analysis: Sentiment-Driven Financial Forecasting

A two-framework pipeline for identifying and tracking financially material ESG drivers using sentiment analysis, robust statistics, and temporal regime detection.

## Overview

This repository implements the methodology described in **[Your Book Title]** for analyzing ESG (Environmental, Social, Governance) materiality through financial market signals.

### Key Features

- **MAD-based Target Creation**: Robust classification of financial performance resistant to outliers and skewness
- **Two-Stage Driver Selection**: Statistical screening (26→12) + Random Forest refinement (12→6) 
- **Temporal Regime Detection**: Identifies shifts in ESG pillar dominance across market cycles
- **SASB Taxonomy Integration**: Maps 26 sustainability drivers to standardized disclosure topics

## Repository Structure

## Repository Structure
│── data/
│   ├── raw/                    # Raw sentiment and financial data
│   └── processed/              # Preprocessed features and targets
├── src/
│   ├── target_engineering.py   # Chapter 4: MAD-based classification
│   ├── feature_selection.py    # Chapter 5: Two-stage driver selection
│   ├── temporal_analysis.py    # Chapter 6: Regime detection
│   └── utils/                  # Helper functions
├── notebooks/
│   ├── 01_target_creation.ipynb
│   ├── 02_driver_selection.ipynb
│   └── 03_regime_tracking.ipynb
├── results/
│   └── entity_analysis/        # JSON outputs per company
├── config/
│   └── parameters.yaml         # Model hyperparameters
└── requirements.txt

## Requirements
```txt
python>=3.8
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
