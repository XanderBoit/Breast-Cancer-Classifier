# Breast-Cancer-Classifier

This notebook demonstrates a machine learning approach for classifying tumors as malignant or benign using the Breast Cancer Wisconsin dataset.

## Model Used
- XGBoost Classifier (`XGBClassifier`)
- Evaluated with accuracy, sensitivity, specificity, and ROC/AUC.

## Dataset
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

## How to Run
1. Install required packages: `pip install -r requirements.txt`
2. Run `Breast_Cancer_Classifier.ipynb` in Jupyter or VSCode.

## Results
- Accuracy: 96.5%
- Sensitivity is 93.3%
- Specificity is 98.2%
- AUC: 0.98

## Visualizations
- **Confusion Matrix**: Highlights correct vs. incorrect predictions
![Confusion Matrix](images/confusion_matrix.png)
- **ROC Curve**: AUC of 0.98 shows strong discriminative power
![ROC Curve](images/ROC.png)
- **Countplot**: Countplot to show distribution of malignant and benign
![SHAP Summary Plot](images/distribution.png)

## License
This project is licensed under the MIT License.
