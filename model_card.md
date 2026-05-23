# Model Card

## Model Details
Alex created the model. It is a Random Forest Classifier trained with max_depth=15, n_estimators=100, and random_state=42 in scikit-learn 1.5.1.

## Intended Use
The intended use of this model is to predict whether an individual's income exceeds $50k, based on census data.

## Training Data
The model was trained on census data from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/20/census+income).
The original dataset has 32561 rows, and an 80-20 split was used to create a training and test set. A One Hot Encoder was used for the categorical features, and a label binarizer was used for the target label.

## Evaluation Data
The evaluation dataset was created by splitting out 20% of the original dataset.

## Metrics
The metrics collected are:
Precision: 0.7918
Recall: 0.5786
F1-Score: 0.6686

## Ethical Considerations
The census data was collected in 1994 and is extremely likely to include representational bias and missing data, which are likely to affect prediction outcomes, both when applied to a historical context and even more so when applied to a modern context.

## Caveats and Recommendations
The census data has not been adjusted for inflation and is purely in 1994 values. 
Further improvements to the model could be achieved through parameter tuning.
All results should be considered for educational purposes only.