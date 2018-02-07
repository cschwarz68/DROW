import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
from sklearn import svm
from sklearn.metrics import roc_curve, auc, classification_report
# from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.model_selection import GridSearchCV

MAKEPLOTS = True
# TARGET can be: 'ord','state'
TARGET = 'ord'


def readfile(numreps):
    features = pd.read_csv('features' + str(numreps) + '.csv',
                           error_bad_lines=False)

    targets = features[['ord', 'state']]
    targets = targets[TARGET]
    del features['ord']
    del features['state']

    return features, targets


def multiple_svm(X, y):
    if TARGET == 'ord':
        thresholdvec = [55, 60, 65, 70]
        n_splits = 4
    elif TARGET == 'state':
        yclass = y.str.contains('NOT')
        y = [1 if c else 0 for c in yclass]
        y = pd.Series(y)
        thresholdvec = [0.5, 0.5]
        n_splits = 2
    else:
        raise ValueError('A bad value for the target variable was used: ',
                         TARGET)

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=n_splits)
    classifier = svm.SVC(kernel='linear', probability=True, random_state=0)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue',
                    'darkorange'])
    lw = 2
    plt.figure()
    i = 0
    for (train, test), color, thresh in zip(cv.split(X, y), colors,
                                            thresholdvec):
        probas_ = classifier.fit(X.loc[train, :],
                                 y[train] < thresh).predict_proba(X.loc[test,
                                                                        :])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test] < thresh, probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color,
                 label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
             label='Luck')

    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def multiple_roc(X, y, classifier):
    if TARGET == 'ord':
        thresholdvec = [55, 60, 65, 70]
        n_splits = 4
    elif TARGET == 'state':
        yclass = y.str.contains('NOT')
        y = [1 if c else 0 for c in yclass]
        y = pd.Series(y)
        thresholdvec = [0.5, 0.5]
        n_splits = 2
    else:
        raise ValueError('A bad value for the target variable was used: ',
                         TARGET)

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=n_splits)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue',
                    'darkorange'])
    lw = 2
    plt.figure()
    i = 0
    for (train, test), color, thresh in zip(cv.split(X, y), colors,
                                            thresholdvec):
        # print np.where(y[train]<thresh)
        # print np.where(y[test]<thresh)
        probas_ = classifier.fit(X.loc[train, :],
                                 y[train] < thresh).predict_proba(X.loc[test,
                                                                        :])
        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(y[test] < thresh, probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color,
                 label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
             label='Luck')

    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    plt.close('all')

    numreps = 10
    features, targets = readfile(numreps)
    y = targets < 70
    X_train, X_test, y_train, y_test = train_test_split(features, y,
                                                        test_size=0.33,
                                                        stratify=y,
                                                        random_state=9)

    params = {'n_estimators': 300, 'max_depth': 3, 'subsample': 1.0,
              'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
    clf = GradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)
    acc = clf.score(features, targets)  # should use test set
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

    params1 = [{'n_estimators': [10, 20],
                'max_depth': [1, 3]}]
    params2 = [{'n_estimators': [10, 20],
                'max_depth': [1, 3],
                'max_features': [None, 'sqrt']}]
    params3 = [
        {'n_estimators': [10, 20, 50, 100, 500],
         'max_depth': [1, 3, 5, 10],
         'max_features': [None, 'sqrt']},
        ]
    # clf = GridSearchCV(classifier, params1, cv=5, scoring='roc_auc',
    #                    n_jobs=-1)
    # s = clf.fit(features, targets)
    # # scores = cross_val_score(classifier, features, targets, cv=5,
    # #                          scoring='roc_auc')
    #
    # print("Best parameters set found on development set:")
    # print()
    # print(clf.best_params_)
    # print()
    # print("Grid scores on development set:")
    # print()
    # means = clf.cv_results_['mean_test_score']
    # stds = clf.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))
    # print()

    # print("Detailed classification report:")
    # print()
    # print("The model is trained on the full development set.")
    # print("The scores are computed on the full evaluation set.")
    # print()
    # y_true, y_pred = y_test, clf.predict(X_test)
    # print(classification_report(y_true, y_pred))
    # print()

    plt.pause(1)
