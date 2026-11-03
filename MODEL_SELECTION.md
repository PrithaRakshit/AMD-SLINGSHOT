Model Selection: Isolation Forest

Isolation Forest was selected for Grid-Guard because the project focuses on identifying unusual observations in power-consumption data.

Unlike conventional supervised classification algorithms, Isolation Forest does not require labelled training examples to learn the normal data distribution.

The algorithm isolates observations using randomly selected features and split values. Anomalous observations are generally isolated in fewer partitioning steps than normal observations.

Reasons for Selection

1. It supports unsupervised anomaly detection.
2. It is suitable for numerical datasets.
3. It provides anomaly predictions and decision scores.
4. It offers a practical starting point for experimentation with synthetic power-consumption data.

Current Scope

The model is being explored on synthetic data. Its effectiveness on real-world power-consumption datasets has not yet been established.

Future Evaluation

Future work can compare Isolation Forest with Local Outlier Factor, One-Class SVM and other anomaly detection approaches using consistent evaluation datasets and metrics.
