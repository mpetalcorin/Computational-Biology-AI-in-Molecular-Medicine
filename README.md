# Computational-Biology-AI-in-Molecular-Medicine
A curated collection of Jupyter notebooks, datasets, and ML models applying bioinformatics, statistics, and AI to real-world biomedical problems—spanning cancer, diabetes, blood pressure, protein-ligand binding, and clinical trials. Designed for research, teaching, and diagnostics.
# Computational Biology of Molecular Systems

This repository contains a collection of Jupyter notebooks, datasets, and machine learning models for exploring applications of computational biology in disease modeling, diagnosis, treatment prediction, and biomedical research.

## Repository Overview

### Notebooks
| Notebook | Description |
|---------|-------------|
| `BLAST_Genetic_Disorder_Diagnosis.ipynb` | Uses BLAST for diagnosing genetic disorders from DNA sequences. |
| `cancer_survival_analysis_realworld.ipynb` | Survival analysis using real-world clinical datasets. |
| `clinical_trial_analysis.ipynb` | Statistical evaluation of clinical trial data. |
| `diabetes_health_indicators_analysis.ipynb` | Exploratory data analysis of diabetes indicators. |
| `diabetes_model_balanced_with_evaluation.ipynb` | Balanced ML model with performance metrics for diabetes prediction. |
| `genotype_to_phenotype_kaggle_demo.ipynb` | Predicts phenotype from genotype using a Kaggle dataset. |
| `hypothesis_testing_biomedical.ipynb` | Introductory hypothesis testing in biomedical datasets. |
| `METABRIC_genotype_phenotype_modeling_Part3.ipynb` | End-to-end modeling using METABRIC data with SHAP and XGBoost. |
| `Predicting_Protein–Ligand_Binding_Affinity_Interpretable_ML.ipynb` | Uses interpretable machine learning for protein-ligand binding affinity. |
| `statistical_tests_blood_pressure.ipynb` | Statistical comparison of treatments affecting blood pressure. |

### Data Files
- `METABRIC_RNA_Mutation.csv` — METABRIC dataset for modeling breast cancer subtypes.
- `blood_pressure_treatment_data.csv` — Sample dataset for blood pressure experiments.
- `diabetes_binary_health_indicators_BRFSS2015.csv` — CDC public health diabetes data.
- `vitamin_d_cold_study.csv` — Nutritional study dataset for vitamin D and cold resistance.

### Learning Resources
- `diabetes_prediction_student_worksheet.txt` — Guided worksheet for students.
- `diabetes_prediction_teachers_guide.txt` — Teaching guide with discussion points.

### Model Files
- `optimized_xgb_model.pkl` — Optimized and exportable XGBoost model from METABRIC notebook.

## Highlights

- **Machine Learning for Disease Modeling** — Random Forest and XGBoost applied to diabetes and breast cancer data.
- **Feature Importance Analysis** — SHAP visualizations and classic impurity-based importance plots.
- **Cross-validation & Grid Search** — Hyperparameter tuning and robust model performance validation.
- **Biomedical Statistics** — From basic hypothesis testing to clinical trial analytics.
- **Protein-Ligand Interaction Prediction** — Interpretable ML in molecular docking scenarios.

## Getting Started

### Prerequisites
Install the required Python libraries:
bash
pip install -r requirements.txt
jupyter lab

### Contributing
This is an educational and research-focused repository. Contributions and collaborations are welcome, especially from those developing new molecular diagnostics or interested in curriculum design for biomedical computing.

### License

MIT License — free to use with attribution.

### Maintainer

Dr Mark Petalcorin — Computational Biology & AI in Molecular Medicine
