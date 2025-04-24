# Best-Model-for-Diabetes-Prediction
The goal of the project is to determine the best model and features to predict if a person has diabetes or not based on  the diagnostic measurements in the dataset.
Ran the following algorithms with hyperparameter tuning to determine best model:  KNN Classifier, Logistic Regression, SVC, and Random Forest. 

Dataset used:  https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset

The process for determining the best model included:

	Data Overview
	EDA – Statistical Summary and Inference
	EDA – Class Label Imbalance
	EDA - Univariate Analysis and Imputing Zero Values
	EDA – Feature Correlation
	EDA – Feature Selection
	Run models with all features and selected features
	Hyperparameter tuning of algorithms to get best accuracy, precision and recall 
	Algorithms Run:
		kNN Classifier: Tuning Hyperparameter k iteratively (n neighbors)
		Logistic Regression: Tuning hyperparameter C iteratively (Inverse of regularization strength)
		SVC: Hyperparameter Tuning for best C and gamma using GridSearchCV
		Random Forest: Hyperparameter tuning iteratively for N (number of trees) and d(max depth )
	Result Analysis 
	Conclusion 

