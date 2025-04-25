import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score,f1_score,
                precision_score, recall_score,classification_report)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_k_for_best_recall_and_accuracy(X_, Y_, X_test_, Y_test_):
    """
    Iterating KNN model with k values from 1 tp 30 to get best k value with for
    maximun recall and accuracy. Plot the accuracy vs recall graph against k
    :param X_: X Train array
    :param Y_: Y train array
    :param X_test_: X test array
    :param Y_test_: Y test array
    :return: best k value
    """
    accuracy_list = []
    recall_list = []
    for k in range(1, 30):
        # create a classifier object
        cls_kNN = KNeighborsClassifier(n_neighbors=k)
        Y_ = np.ravel(Y_)
        cls_kNN.fit(X_, Y_)
        # Predict the classes of the data in the test set:
        Y_pred = cls_kNN.predict(X_test_)
        # calculate  and print accuracy
        accuracy = round(accuracy_score(Y_test_, Y_pred), 3)
        accuracy_list.append(accuracy)
        recall = round(recall_score(Y_test_, Y_pred), 3)
        recall_list.append(recall)
    max_ac_index = accuracy_list.index(max(accuracy_list))
    max_re_index = recall_list.index(max(recall_list))
    # print("Values for max accuracy: k max_accuracy, recall_at_max_Accuracy\n",
    #       max_ac_index + 1, max(accuracy_list), recall_list[max_ac_index])
    if recall_list[max_ac_index] >= max(recall_list):
        best_k = max_ac_index + 1
    else:
        best_k = max_re_index + 1

    # print('best_k:', best_k)
    # plot k vs accuracy plot to get best k
    fig, axs = plt.subplots(1, 1)
    axs.plot(range(1, 30), accuracy_list, color="blue", label='accuracy')
    axs.plot(range(1, 30), recall_list, color="red", label='recall')
    axs.legend()
    fig.set_size_inches(10, 7)
    axs.set_xlabel("k value")
    axs.set_ylabel("Accuracy and Recall")
    axs.set_title("KNN Classification : Accuracy vs Recall plot")
    # showing and saving plot
    fig.show()
    fig.savefig(input_dir + "\\kNN_accuracy_plot.png")
    # for best k = 18, model the kNN classifier
    return best_k


def impute_feature(df_, var_, type_):
    """
    imputing zero values in the var_ column with mean or median
    :param df_: dataframe where values are to be imputed
    :param var_: feature  in df whose values are to be imputed
    :param type_: mean - impute with mean , median - impute with median
    :return: modified dataframe
    """
    # finding medain insulin for Outcome = 1 and Outcome = 0
    if type_ == "mean":
        impute_val = df_[var_].mean()
    elif type_ == "median":
        impute_val = df_[var_].median()
    # replacing Nan Values with median values for the outcome
    # print(df.fillna({'name': 'XXX', 'age': 20, 'ZZZ': 100}))
    df_.fillna({var_: impute_val}, inplace=True)
    return df_


def train_split_and_scale(df_, scale_, X_columns_, Y_columns_, test_size_):
    """
    splits the data to train and test depending on test size and stratify_
    :param X_:  numpy features array
    :param Y_: numpy class(label) array
    :param scale_: 1 if scaling needed else no scaling
    :param test_size_: percent of testing data
    :param stratify_: Column on which to stratify - usually class label
    :return: list of [X_train,X_test,Y_train,Y_test]
    """
    X_ = df_[X_columns_]
    Y_ = df_[Y_columns_].to_numpy()

    X_train_, X_test_, Y_train_, Y_test_ = train_test_split(X_,
                                                            Y_,
                                                            test_size=test_size_,
                                                            random_state=1,
                                                            stratify=Y_)
    if scale_ == 1:
        scaler = StandardScaler()
        X_train_ = scaler.fit_transform(X_train_)
        X_test_ = scaler.transform(X_test_)
    return [X_train_, X_test_, Y_train_, Y_test_]

def get_best_c_val_LR(X_,Y_,X_test_,Y_test_):
    """
    get best c value for Logistic Regression by running the model iteratively
    for instance with the highest accuracy.
    :param X_: X_train
    :param Y_: Y_train
    :param X_test_: X_test
    :param Y_test_: Y_test
    :return: best C value
    """
    c_list  = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    max_accuracy = 0
    opt_c = 0
    Y_ = np.ravel(Y_)
    for c_val in c_list:
        # Create a logistic regression model
        lr_class = LogisticRegression(C=c_val,class_weight='balanced')
        # Fit the model on the scaled training data
        lr_class.fit(X_, Y_)
        # Make predictions on the testing data
        y_predicted = lr_class.predict(X_test_)
        accuracy = accuracy_score(Y_test_, y_predicted)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            opt_c = c_val
    return opt_c
def compute_performance_measures(y_test_, y_pred_, classifier_type_,
                                 classifier_):
    """
    Computes and returns the performance measures for the predicted and
    actual values including confusion matrix. Plots the confusion martix for
    the same
    :param y_test_: Actual Test Class Label
    :param y_pred_:  Predicted Class Label
    :param classifier_type_: Classifier Type : NB, DT or Random Forest
    :return: None
    """
    cm = confusion_matrix(y_test_, y_pred_)
    # print(classifier_type_.strip(), ": Confusion Matrix:\n", cm)
    tn, fp, fn, tp = cm.ravel()
    tpr = round(tp / (tp + fn), 4)
    tnr = round(tn / (tn + fp), 4)
    accuracy = round(accuracy_score(y_test_, y_pred_),4)

    precision = round(precision_score(y_test_, y_pred_),4)
    f_score = round(f1_score(y_test_,y_pred_),4)
    print(classifier_type_.strip())
    print("-" * 50)
    # print(f'Hyperparameters:'
    #       f'{classifier_.get_params()}')
    print(f'TN       FP        FN        TP             Accuracy          '
          f'Precision                Recall                 f1_score        '
          f'TNR')
    print(f'{tn}       {fp}        {fn}        {tp}             {accuracy}  '
          f'            {precision}             {tpr}              {f_score}    '
          f'   {tnr}     ')

    # print("TN:",tn)
    # print("FP:", fp)
    # print("FN:", fn)
    # print("TP:", tp)
    # print("Accuracy:", accuracy)
    # print("Precision:", precision)
    # print("Recall:",tpr)
    # print("f1_score:",f_score)
    # print("TNR:",tnr)
    # plotting confusion matrix
    label = "cm: " + classifier_type_
    sns.heatmap(cm, square=True, annot=True, fmt='d', cmap="YlGnBu")
    plt.title(label, fontsize=10)
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    plt.show()


def data_processing_and_eda(df_):
    """
    Perform the data preprocseeing and EDA operations
    :param df_: dataframe with dataset
    :return: None
    """

    # data size and null check
    print("Size of Data: ", df_.shape)
    # statistical information
    print("Statistical Summary: ",df_.describe())
    # class label imbalance count an d plots
    print("Count of class Labels:", df_.groupby("Outcome").size())
    fig = plt.figure(figsize=(6, 6))
    plt.hist(df_["Outcome"], bins=3)
    plt.title("Count of class variable: Outcome")
    plt.xlabel("Outcome")
    plt.xticks(np.arange(0, 2, 1))
    plt.ylabel("Count")
    plt.show()
    # univariate feature histograms
    fig = plt.figure(figsize=(6, 12))
    df_.drop(["Outcome"], axis=1).hist()
    plt.tight_layout()
    plt.show()
    # zero value count by feature
    print("EDA : Count of zeros by column:")
    for cols in df_.columns:
        print(cols, ":", df_.query(f"{cols} == 0").shape[0])
    # multivariate feature analysis: correlation plot
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(df_.corr(), annot=True, cmap="YlGnBu")
    plt.title("Correlation Heatmap for diabetes dataset")
    plt.show()
    return None
def get_best_N_and_D_for_random_forest(X_,Y_,X_test_,Y_test_):
    """
    Get the best N(no of trees) and d (max depth) for random forest model
    iteratively for model with the lowest error. Plot error rate vs number of
    tress for different d values.
    :param X_: X_train
    :param Y_: Y_train
    :param X_test_: X_test
    :param Y_test_: Y_test
    :return: best N , best D
    """
    N = 20  # max number of trees
    d = 6  # max depth
    er_list = []
    Y_ = np.ravel(Y_)
    for i in range(1, N + 1):
        for j in range(1, d + 1):
            rf = RandomForestClassifier(n_estimators=i,
                    max_depth=j, class_weight = "balanced",random_state=3)

            rf.fit(X_, Y_)
            y_pred_rf = rf.predict(X_test_)
            # er_rate = np.mean(y_pred != Y_test)
            er_rate = (1 - accuracy_score(Y_test_, y_pred_rf))
            er_list.append([i, j, er_rate])

    # converting er_list into a dataframe
    df_er = pd.DataFrame(er_list, columns=['N', 'd', 'error_rate'])
    # getting row with minimum error rate
    min_row = df_er.loc[df_er['error_rate'] == df_er['error_rate'].min()]
    min_row = min_row.iloc[0]

    # getting N and d values for minimum error
    best_N = int(min_row.loc["N"])
    best_D = int(min_row.loc["d"])
    # plotting N,d and error_rate to get minimum error
    color_list = ['red', 'blue', 'green', 'orange', 'purple','brown']
    fig = plt.subplot()
    # looping for d = 1 to 5
    for i in range(1, d + 1):

        df_plt = df_er.loc[df_er['d'] == i]
        lbl = "max_depth d = " + str(i)
        plt.plot(df_plt['N'], df_plt['error_rate'],
                 color=color_list[i - 1], label=lbl)
    plt.xticks(np.linspace(1, N, N))
    plt.title("Random Forest: Error rates for different N's and d's")
    plt.xlabel("N:Number of trees")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.show()
    return best_N,best_D

# main program start
my_file = 'diabetes.csv'
input_dir = os.path.abspath(os.path.curdir)
my_file_path = os.path.join(input_dir, my_file)

try:
    ###########################################################################
    # Group 3 features :  MSTV, Width, Mode, Variance
    # Class label = NSP = Normal=1; Suspect=2; Pathologic=3
    # Question 1 - Part 1
    # load the Excel ("raw data" worksheet) data into Pandas
    # dataframe for the four  feature and class NSP above
    ##########################################################################
    # reading data into dataframe
    df = pd.read_csv(my_file_path)
    # set column names in list
    df_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
                  'Outcome']
    # rename column DiabetesPedigreeFunction to DPedFunc for plot
    df.rename({'DiabetesPedigreeFunction': 'DPedFunc'},
              axis=1, inplace=True)

    # preprocessign and eda
    data_processing_and_eda(df)

    # rename column DPedFunc back to DiabetesPedigreeFunction for processing
    df.rename({'DPedFunc': 'DiabetesPedigreeFunction'},
              axis=1, inplace=True)

    # Preparing Data for Imputing by replacing all zeros with null values for
    # columns  with zero values except Pregnancies which can have zero value

    df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age']] = df[
        ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
         'BMI', 'DiabetesPedigreeFunction', 'Age']].replace(0, np.NaN)
    # print("Count Null Values replaing 0 with Nan:",
    #       df.isnull().sum())
    # Running Classifiers for all the features

    ########################################################################
    # Running models for all feature of the dataset
    ########################################################################
    # Running Classifiers for all the features
    print("Running Classifiers for All Features")
    X_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    Y_Columns = ["Outcome"]
    # Splitting Data into Train and Test before imputing
    X_train, X_test, Y_train, Y_test = train_split_and_scale(df, 1,
                       X_columns, Y_Columns, 0.25)

        # Training and testing dataframe for imputing
    df_train_im = pd.DataFrame(X_train)
    df_train_im.columns = X_columns
    df_test_im = pd.DataFrame(X_test)
    df_test_im.columns = X_columns
    cols_to_impute  =  [["Insulin", "median"],["BMI", "mean"],
     ["BloodPressure", "mean"],["SkinThickness", "mean"],["Glucose", "mean"]]

    #imputing training features data
    for col in cols_to_impute:
        df_train_im = impute_feature(df_train_im, col[0], col[1])
    # print("Count of Nulls after imputing for training :\n",
    #       df_train_im.isnull().sum())
    # print(df_train_im.head(),df_train_im.shape)
    # imputing testing features data
    for col in cols_to_impute:
        df_test_im = impute_feature(df_test_im, col[0], col[1])


    X_train_im = df_train_im.to_numpy()
    X_test_im = df_test_im.to_numpy()

    ########################################################################
    # Tuning Knn Classifier hyperparameter k = number of nearest neighbors
    # in range 1 tp 30
    ########################################################################

    best_k = get_k_for_best_recall_and_accuracy(X_train_im,Y_train,X_test_im,
                                                Y_test)
    print("kNN Classifier: best k:",best_k)
    cls_kNN = KNeighborsClassifier(n_neighbors=best_k)
    clf_list = [[cls_kNN, "kNN Classifier('All Features')"]]

    # Tuning hyperparameter C
    # Reducing class imbalance  :  class_weight = balanced
    opt_c = get_best_c_val_LR(X_train_im, Y_train, X_test_im, Y_test)

    #creating LR model for best C  = 1 value
    print("Logistic Regression: best C:", opt_c)
    # Create a logistic regression model
    cls_lr = LogisticRegression(C=opt_c, class_weight="balanced")
    # adding to  list of classifiers
    clf_list.append([cls_lr, "Logistic Regression('All Features')"])


    best_N,best_d= get_best_N_and_D_for_random_forest(X_train_im,Y_train,
                                     X_test_im,Y_test)
    print(f'Random Forest: Best N: {best_N} Best d: {best_d}')
    cls_rf = RandomForestClassifier(n_estimators=best_N, max_depth=best_d,
                        random_state=3,class_weight = "balanced")

    clf_list.append([cls_rf, "Random Forest Classifier('All Features')"])

    # hyperparameter tuning for SVC vis gridsearcnCV
    param_grid = param_grid = { 'C': [0.1,1, 10, 100, 1000],
                  'gamma': [ 1, 0.1, 0.01, 0.001, 0.0001],
                  }


    cls_svc_gs = GridSearchCV(SVC(class_weight="balanced"),
                              param_grid, cv=5, verbose=0)

    clf_list.append([cls_svc_gs, "SVM with GridSearchCV('All Features')"])
    # for each classifier fit on training data and predict on testing data for
    # subset
    Y_train = np.ravel(Y_train)
    for cl in clf_list:

        cl[0].fit(X_train_im, Y_train)
        Y_pred_s = cl[0].predict(X_test_im)
        if (cl[0] == cls_rf):
            feature_scores = pd.Series(cls_rf.feature_importances_  ,
            index=df_train_im.columns).sort_values(ascending=False)
            print("Feature Importances RF "
                  "Classifier:\n", feature_scores)
        elif(cl[0] == cls_svc_gs):
            # print best parameter after tuning
            print("grid best params", cls_svc_gs.best_params_)
            # print how our model looks after hyper-parameter tuning
            print("grid best estimator", cls_svc_gs.best_estimator_)
        compute_performance_measures(Y_test, Y_pred_s, cl[1],cl[0])




    ########################################################################
    # Running models for selected features based on correlation and Random
    # forest feature importance
    # Running the model on highest correlation features:
    # Glucose, BMI, Age and Diabetes Pedigree function
    #Since Pregaancies ans age are highly correlated using only age
    ########################################################################

    # Running Classifiers for selected features

    X_columns = ['Glucose', 'BMI','Age','DiabetesPedigreeFunction']
    print(f'Running Classifier for Selected Features: " {X_columns}')
    Y_Columns = ["Outcome"]
    # Splitting Data into Train and Test before imputing
    X_train, X_test, Y_train, Y_test = train_split_and_scale(df, 1,
                            X_columns, Y_Columns, 0.25)

    # Training and testing dataframe for imputing
    df_train_im = pd.DataFrame(X_train)
    df_train_im.columns = X_columns
    df_test_im = pd.DataFrame(X_test)
    df_test_im.columns = X_columns
    cols_to_impute = [ ["BMI", "mean"], ["Glucose", "mean"]]

    # cols_to_impute = [["Insulin", "median"], ["BMI", "mean"],
    #                   ["BloodPressure", "mean"], ["SkinThickness", "mean"],
    #                   ["Glucose", "mean"]]

    # imputing training features data
    for col in cols_to_impute:
        df_train_im = impute_feature(df_train_im, col[0], col[1])
    # print("Count of Nulls after imputing for training :",
    #       df_train_im.isnull().sum())
    # print(df_train_im.head(), df_train_im.shape)
    # imputing testing features data
    for col in cols_to_impute:
        df_test_im = impute_feature(df_test_im, col[0], col[1])


    X_train_im = df_train_im.to_numpy()
    X_test_im = df_test_im.to_numpy()

    ########################################################################
    # Tuning Knn Classifier hyperparameter k = number of nearest neighbors
    # in range 1 tp 30
    ########################################################################

    best_k = get_k_for_best_recall_and_accuracy(X_train_im, Y_train, X_test_im,
                                                Y_test)
    print("kNN Classifier: best k: ", best_k)
    cls_kNN = KNeighborsClassifier(n_neighbors=best_k)
    clf_list = [[cls_kNN, "kNN Classifier(Selected Features)"]]

    # Tuning hyperparameter C
    # Reducing class imbalance  :  class_weight = balanced
    opt_c = get_best_c_val_LR(X_train_im, Y_train, X_test_im, Y_test)

    # creating LR model for best C  = 1 value
    print("Logistic Regression: best C:", opt_c)
    # Create a logistic regression model
    cls_lr = LogisticRegression(C=opt_c, class_weight="balanced")
    # adding to  list of classifiers
    clf_list.append([cls_lr, "Logistic Regression(Selected Features)"])

    best_N, best_d = get_best_N_and_D_for_random_forest(X_train_im, Y_train,
                                                        X_test_im, Y_test)
    print(f'Random Forest: Best N: {best_N} Best d: {best_d}')
    cls_rf = RandomForestClassifier(n_estimators=best_N, max_depth=best_d,
                                    random_state=3, class_weight="balanced")

    clf_list.append([cls_rf, "Random Forest Classifier(Selected Features)"])

    # hyperparameter tuning for SVC vis gridsearcnCV
    param_grid = {'C': [0.1,1, 10, 100, 1000],
                  'gamma': [ 1, 0.1, 0.01, 0.001, 0.0001],
                  }


    cls_svc_gs = GridSearchCV(SVC(class_weight="balanced"),
                              param_grid, cv=5, verbose=0)

    clf_list.append([cls_svc_gs,"Best SVM with GridSearchCV(Selected "
                                "Features)"])
    # for each classifier fit on training data and predict on testing data for
    # subset
    Y_train = np.ravel(Y_train)
    for cl in clf_list:

        cl[0].fit(X_train_im, Y_train)
        Y_pred_s = cl[0].predict(X_test_im)
        if (cl[0] == cls_rf):
            feature_scores = pd.Series(cls_rf.feature_importances_,
                                       index=df_train_im.columns).sort_values(
                ascending=False)
            print("Feature Importances RF "
                  "Classifier:\n", feature_scores)
        elif (cl[0] == cls_svc_gs):
            # print best parameter after tuning
            print("grid best params", cls_svc_gs.best_params_)
            # print how our model looks after hyper-parameter tuning
            print("grid best estimator", cls_svc_gs.best_estimator_)
        compute_performance_measures(Y_test, Y_pred_s, cl[1], cl[0])





except Exception as f:
    print(f)
    print('Error during program execution while processing file: ', my_file)
