import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
from sklearn import svm
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.model_selection import GridSearchCV

MAKEPLOTS = True
# TARGET can be: 'ord','state','type'
TARGET = 'ord'


def readfile(numreps):
    numreps = int(numreps)
    features = pd.read_csv('features' + str(numreps) + '.csv',
                           error_bad_lines=False)

    eventid = pd.DataFrame(features['eventid'])
    ordval = pd.DataFrame(features['ord'])

    targets = features[['ord', 'state', 'type']]
    targets = targets[TARGET]
    del features['ord']
    del features['state']
    del features['type']
    del features['eventid']

    return features, targets, eventid, ordval


def search_train(clf, params, X_train, y_train, X_test, y_test):
    clfgrid = GridSearchCV(clf, params1, cv=5, scoring='roc_auc', n_jobs=-1)
    clfgrid.fit(X_train, y_train)
    clfgrid.best_score_
    clfbest = clfgrid.best_estimator_

    # Compute ROC curve and area the curve
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    probas_ = clfbest.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='area = %0.2f' % (roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Luck')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    importances = clfbest.feature_importances_
    if "GradientBoostingClassifier" in str(type(clf)):
        std = np.std([tree[0].feature_importances_
                      for tree in clfbest.estimators_], axis=0)
    else:
        std = np.std([tree.feature_importances_
                      for tree in clfbest.estimators_], axis=0)

    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, X_train.columns[indices[f]],
                                       importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.xlabel('Feature ID')
    plt.ylabel('Importance')
    # plt.title('Receiver operating characteristic example')
    plt.show()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clfbest.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    print("Matthews correlation coefficient")
    print(matthews_corrcoef(y_true, y_pred))
    print("AUC")
    print(roc_auc)
    print("confusion matrix")
    print(confusion_matrix(y_true, y_pred))

    pred = pd.Series(y_pred, index=y_true.index)
    probas = pd.DataFrame(probas_, index=y_true.index)
    output = pd.concat([y_true, pred, probas], axis=1)
    output.columns = ['truth', 'prediction', 'prob0', 'prob1']
    return output


if __name__ == '__main__':
    plt.close('all')

    params = {'n_estimators': 10, 'random_state': 3}
    clf1 = GradientBoostingClassifier(**params)
    clf2 = RandomForestClassifier(**params)
    clf3 = ExtraTreesClassifier(**params)
    clf4 = AdaBoostClassifier(**params)

    numreps = 10
    features, targets, eventid, ordval = readfile(numreps)

    if TARGET == 'ord':
        # use ORD of 50 as classification threshold
        y = targets < 50
        X_train, X_test, y_train, y_test = train_test_split(features, y,
                                                            test_size=0.33,
                                                            stratify=y,
                                                            random_state=9)
        params1 = [{'n_estimators': [10, 50, 100, 200, 500, 700]}]
        output = search_train(clf3, params1, X_train, y_train, X_test, y_test)
        output = pd.concat([eventid, X_test, ordval, output], axis=1,
                           join='inner')
        output.sort_index(inplace='True')
        output.to_csv('final_output_ord50.csv', index=None)

        # use ORD of 70 as classification threshold
        y = targets < 70
        X_train, X_test, y_train, y_test = train_test_split(features, y,
                                                            test_size=0.33,
                                                            stratify=y,
                                                            random_state=9)
        output = search_train(clf1, params1, X_train, y_train, X_test, y_test)
        output = pd.concat([eventid, X_test, ordval, output], axis=1,
                           join='inner')
        output.sort_index(inplace='True')
        output.to_csv('final_output_ord70.csv', index=None)
    elif TARGET == 'type':
        y = pd.Series([True if 'Baseline' in sub else False for sub in
                       targets])
        X_train, X_test, y_train, y_test = train_test_split(features, y,
                                                            test_size=0.33,
                                                            stratify=y,
                                                            random_state=9)
        params1 = [{'n_estimators': [10, 50, 100, 200, 500, 700]}]
        search_train(clf3, params1, X_train, y_train, X_test, y_test)
    elif TARGET == 'state':
        y = pd.Series([True if 'NOT' in sub else False for sub in targets])
        X_train, X_test, y_train, y_test = train_test_split(features, y,
                                                            test_size=0.33,
                                                            stratify=y,
                                                            random_state=9)
        params1 = [{'n_estimators': [10, 50, 100, 200, 500, 700]}]
        search_train(clf3, params1, X_train, y_train, X_test, y_test)
    else:
        print('unrecognized value for TARGET: ', TARGET)


    plt.pause(1)