import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
from sklearn import svm
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier)
from sklearn.model_selection import GridSearchCV

MAKEPLOTS = True
#NUMREPS = 20
FILENAME = 'combined300.csv'
# TARGET can be: 'ord','state'
TARGET = 'ord'

def readfile(numreps):
    features = pd.read_csv(FILENAME, 
        usecols=['eventid','gyro_z_25%','gyro_z_50%','gyro_z_75%',
                 'gyro_z_std','headway_25%',
                 'lanepos_25%','lanepos_50%','lanepos_75%',
                 'lanepos_std','ord','speed_25%','speed_50%','speed_75%',
                 'speed_std','time_bin','type','state','laneslope'],
        error_bad_lines=False)
    print features.shape
    features[features['headway_25%']<20] = np.nan
    features.drop(['headway_25%'],axis=1,inplace=True)
    print features.shape
    features[features['speed_50%']<20] = np.nan
    print features.shape
    features.dropna(axis=0,how='any',inplace=True)
    print features.shape
    features.reset_index(drop=True, inplace=True)
    
    grouped = features.groupby('eventid')
    
    counts = grouped.apply(lambda x: len(x))
    plt.figure(1)
    counts.plot.hist()
    plt.show()
    
    ordrating = grouped.apply(lambda x: max(x.ord))
    plt.figure(2)
    ordrating.plot.hist()
    plt.show()
    
    df_features = grouped.apply(input_features,numreps)
    df_features.reset_index(inplace=True, drop=True)
    df_targets = df_features[TARGET]
    del df_features[TARGET]
    
    return df_features, df_targets

def input_features(group,numreps):
    # return nothing if too few rows in the group
    if len(group)<numreps:
        return

    # indices to pull features from
    #idx = np.linspace(0, len(group)-1, num=numreps).astype(int)
    idx = np.linspace(len(group)-numreps, len(group)-1, num=numreps).astype(int)
    
    # construct column names
    varnames = ['gyro_z_50%','lanepos_50%','laneslope','speed_50%']
    columns = [v+'_'+str(i) for v in varnames for (i,ix) in enumerate(idx)]
    columns.append('laneslope_75%')
    columns.append('gyro_z_25%_25%')
    columns.append('gyro_z_75%_75%')
    columns.append('lanepos_25%_25%')
    columns.append('lanepos_75%_75%')
    columns.append('time_bin')
    columns.append(TARGET)
        
    # take magnitude of the laneslope
    group.loc[:,'laneslope'] = abs(group['laneslope'])
    
    # summary stats of group
    df = group.describe()
    
    # add additional features to the vector
    group.reset_index(drop=True, inplace=True)
    feature_array = group.loc[idx,varnames].values.T.ravel()
    features = np.append(feature_array,
        [df['laneslope']['75%'], 
        df['gyro_z_25%']['25%'], df['gyro_z_75%']['75%'], 
        df['lanepos_25%']['25%'], df['lanepos_75%']['75%'], 
        max(group['time_bin']), 
        max(group[TARGET])])
    
    df_features = pd.DataFrame(features).T
    df_features.columns = columns

    return df_features
    
def someplots(df):
    F = plt.figure()
    plt.subplot(321)
    plt.plot(df.timestamp,df.lanepos)
    plt.title('lanepos')
    
def multiple_svm(X,y):
    if TARGET == 'ord':
        thresholdvec = [55,60,65,70]
        n_splits = 4
    elif TARGET == 'state':
        yclass = y.str.contains('NOT')
        y = [1 if c else 0 for c in yclass]
        y = pd.Series(y)
        thresholdvec = [0.5, 0.5]
        n_splits = 2
    else:
        raise ValueError('A bad value for the target variable was used: ', TARGET)
    
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=n_splits)
    classifier = svm.SVC(kernel='linear', probability=True, random_state=0)
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    lw = 2
    plt.figure()
    i = 0
    for (train, test), color, thresh in zip(cv.split(X, y), colors, thresholdvec):
        probas_ = classifier.fit(X.loc[train,:], y[train]<thresh).predict_proba(X.loc[test,:])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test]<thresh, probas_[:, 1])
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

def multiple_roc(X,y,classifier):
    if TARGET == 'ord':
        thresholdvec = [55,60,65,70]
        n_splits = 4
    elif TARGET == 'state':
        yclass = y.str.contains('NOT')
        y = [1 if c else 0 for c in yclass]
        y = pd.Series(y)
        thresholdvec = [0.5, 0.5]
        n_splits = 2
    else:
        raise ValueError('A bad value for the target variable was used: ', TARGET)
    
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=n_splits)
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    lw = 2
    plt.figure()
    i = 0
    for (train, test), color, thresh in zip(cv.split(X, y), colors, thresholdvec):
        #print np.where(y[train]<thresh)
        #print np.where(y[test]<thresh)
        probas_ = classifier.fit(X.loc[train,:], y[train]<thresh).predict_proba(X.loc[test,:])
        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(y[test]<thresh, probas_[:, 1])
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

    # takes a LONG time
    #classifier = svm.SVC(kernel='linear', probability=True,
    #                    random_state=0)
    #multiple_roc(features,targets,classifier)

    NUMESTIMATORS = 500
    
    numreps = 20
    features,targets = readfile(numreps)
        
    #classifier = RandomForestClassifier(max_depth=3, n_estimators=NUMESTIMATORS)
    #multiple_roc(features,targets,classifier)
    #
    #classifier = ExtraTreesClassifier(n_estimators=NUMESTIMATORS, max_depth=None,
    #    min_samples_split=2, random_state=0)
    #multiple_roc(features,targets,classifier)
    
    classifier = GradientBoostingClassifier(n_estimators=NUMESTIMATORS, 
        learning_rate=1.0,max_depth=1, random_state=0)
    multiple_roc(features,targets,classifier)
    
    #classifier = AdaBoostClassifier(n_estimators=NUMESTIMATORS)
    #multiple_roc(features,targets,classifier)
    
    numreps = 30
    features,targets = readfile(numreps)
    
    classifier = RandomForestClassifier(max_depth=3, n_estimators=NUMESTIMATORS)
    multiple_roc(features,targets,classifier)
    
    classifier = ExtraTreesClassifier(n_estimators=NUMESTIMATORS, max_depth=None,
        min_samples_split=2, random_state=0)
    multiple_roc(features,targets,classifier)
    
    classifier = GradientBoostingClassifier(n_estimators=NUMESTIMATORS, 
        learning_rate=1.0,max_depth=1, random_state=0)
    multiple_roc(features,targets,classifier)
    
    classifier = AdaBoostClassifier(n_estimators=NUMESTIMATORS)
    multiple_roc(features,targets,classifier)

#    numreps = 40
#    features,targets = readfile(numreps)
#    
#    classifier = RandomForestClassifier(max_depth=3, n_estimators=NUMESTIMATORS)
#    multiple_roc(features,targets,classifier)
#    
#    classifier = ExtraTreesClassifier(n_estimators=NUMESTIMATORS, max_depth=None,
#        min_samples_split=2, random_state=0)
#    multiple_roc(features,targets,classifier)
#    
#    classifier = GradientBoostingClassifier(n_estimators=NUMESTIMATORS, 
#        learning_rate=1.0,max_depth=1, random_state=0)
#    multiple_roc(features,targets,classifier)
#    
#    classifier = AdaBoostClassifier(n_estimators=NUMESTIMATORS)
#    multiple_roc(features,targets,classifier)
#
#    numreps = 50
#    features,targets = readfile(numreps)
#    
#    classifier = RandomForestClassifier(max_depth=3, n_estimators=NUMESTIMATORS)
#    multiple_roc(features,targets,classifier)
#    
#    classifier = ExtraTreesClassifier(n_estimators=NUMESTIMATORS, max_depth=None,
#        min_samples_split=2, random_state=0)
#    multiple_roc(features,targets,classifier)
#    
#    classifier = GradientBoostingClassifier(n_estimators=NUMESTIMATORS, 
#        learning_rate=1.0,max_depth=1, random_state=0)
#    multiple_roc(features,targets,classifier)
#    
#    classifier = AdaBoostClassifier(n_estimators=NUMESTIMATORS)
#    multiple_roc(features,targets,classifier)

    numreps = 30
    features,targets = readfile(numreps)

    classifier = GradientBoostingClassifier(n_estimators=10, 
        learning_rate=1.0,max_depth=1, random_state=0)
    multiple_roc(features,targets,classifier)
    classifier = GradientBoostingClassifier(n_estimators=20, 
        learning_rate=1.0,max_depth=1, random_state=0)
    multiple_roc(features,targets,classifier)
    classifier = GradientBoostingClassifier(n_estimators=40, 
        learning_rate=1.0,max_depth=1, random_state=0)
    multiple_roc(features,targets,classifier)
    classifier = GradientBoostingClassifier(n_estimators=100, 
        learning_rate=1.0,max_depth=1, random_state=0)
    multiple_roc(features,targets,classifier)
    classifier = GradientBoostingClassifier(n_estimators=200, 
        learning_rate=1.0,max_depth=1, random_state=0)
    multiple_roc(features,targets,classifier)
    classifier = GradientBoostingClassifier(n_estimators=350, 
        learning_rate=1.0,max_depth=1, random_state=0)
    multiple_roc(features,targets,classifier)
    
    tuned_parameters = [
        {'n_estimators': [10,20,50,100,500], 'max_depth': [3,5,10], 'max_features': [None,'sqrt']},
        ]
    clf = GridSearchCV(classifier, tuned_parameters, cv=5, scoring='roc_auc', n_jobs=-1)
    clf.fit(features, targets)
    #scores = cross_val_score(classifier, features, targets, cv=5, scoring='roc_auc')
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    #print()
    #print("The model is trained on the full development set.")
    #print("The scores are computed on the full evaluation set.")
    #print()
    #y_true, y_pred = y_test, clf.predict(X_test)
    #print(classification_report(y_true, y_pred))
    #print()


    plt.pause(1)