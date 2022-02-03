import os
import json
import time as t
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance
import scipy.stats

from sklearn import svm, datasets, metrics, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import KBinsDiscretizer

def drop_correlated(dataframe, direction, threshold=0.9):
    to_drop = []
    to_keep = []
    matrix = abs(dataframe.corr().values)
    featureList = dataframe.columns

    if direction == '>':
        print("Dropping correlated features greater than ", threshold)
        # Get max and max_i of each row
        for i in range(matrix.shape[0]-1): # exclude label
            # Drop diagonal element, which is = 1
            row = np.delete(matrix[i,:], i, axis=0)
            # Zeroize nan values
            row[np.isnan(row)] = 0
            if row.shape[0] > 0:
                maximum = np.amax(row)
                max_i = np.argmax(row)
                # Add to to_drop and to_keep
                if maximum == 0 or (maximum > threshold and max_i not in to_keep):
                    print("Dropping {}. Correlation with {}: {}".format(featureList[i], featureList[max_i], maximum))
                    to_drop.append(featureList[i])
                    to_keep.append(max_i)
            else:
                print("{} is a all-nan feature. Dropping".format(featureList[i]))
                if featureList[i] not in to_drop:
                    to_drop.append(featureList[i])
    elif direction == '<':
        print("Dropping features uncorrelated to target less than ", threshold)
        row = np.nan_to_num(matrix[-1,:-1]) # exclude label itself
        for i in range(len(row)): 
            if row[i] < threshold:
                print("Dropping {}. Correlation with target: {}".format(featureList[i], row[i]))
                to_drop.append(featureList[i])
    else:
        ValueError("Invalid direction. Options: < , > ")


    dataframe = dataframe.drop(to_drop, axis=1)
    print("{} features were dropped. New feature size: {}".format(len(to_drop), dataframe.values.shape[1]-1))
    return dataframe


def count(array, i, bin_start):
    count = 0
    for element in array:
        if element >= bin_start[i] and element < bin_start[i+1]:
            count += 1

    return count


def get_distributions(x, y, n_bins=100):
    epsilon = 1e-5 # added to maximum in order to be inclusive for the max value.

    minimum = min(x.min(), y.min())
    maximum = max(x.max(), y.max()) + epsilon

    bin_start = np.linspace(minimum, maximum, n_bins+1)

    x_dist = [count(x, i, bin_start) for i in range(n_bins)]
    y_dist = [count(y, i, bin_start) for i in range(n_bins)]

    return np.asarray(x_dist), np.asarray(y_dist)
    

def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance 
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance


def FSMJ(dataframe, target, n_bins=100): # Feature Selection based on Mean Jensen-Shannon Divergence
    # for each feature, get separate arrays for each class
    mean_distances = []
    n_classes = int(target.values.max())+1
    
    with open("./distances.csv", "w") as f:
        f.write("idx,featurename,")
        for m in range(n_classes):
            for l in range(n_classes-m-1):
                f.write("{}-{},".format(m, m+l+1))
        f.write("\n")

        for i, feature in enumerate(dataframe.columns):
            f.write("{},{},".format(i, feature))
            array_list = []
            distances = []
            # for each class, extract feature arrays
            for l in range(n_classes):
                is_l = target.values == l
                array_list.append(dataframe.loc[is_l].values[:,i]) 
                if l > 0:
                    p, q = get_distributions(array_list[0], array_list[l], n_bins)
                    #distances.append(jensen_shannon_distance(p, q))
                    d = distance.jensenshannon(p, q, 2) # base:2 makes max=1.0
                    distances.append(d)
                    f.write("{:3f},".format(d))
            for m in range(1,n_classes):
                for l in range(n_classes-m-1):
                    p, q = get_distributions(array_list[m], array_list[m+l+1], n_bins)
                    #distances.append(jensen_shannon_distance(p, q))
                    d = distance.jensenshannon(p, q, 2) # base:2 makes max=1.0
                    distances.append(d)
                    f.write("{:3f},".format(d))                    

            mean_distances.append(np.mean(distances))
            print(i, feature, mean_distances[i])
            f.write("\n")

    return mean_distances


def one_hot(y_, n_classes=7):
    # Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


def write_featureDict(datasetFolderName, TLS=False, DNS=False, HTTP=False):
    featureDict = {}
    for root, dirs, files in os.walk(datasetFolderName):   
        for f in files:                        
            if f.endswith((".json")):     
                # Open json file
                print("Processing ", f)
                with open(os.path.join(root, f), "r") as jj:
                    #data = [json.loads(line) for line in jj]
                    # Write a for loop and check every single flow with utf-8:
                    data = []
                    i = 0
                    while True:
                        i += 1
                        try:
                            flow = jj.readline()
                            if not flow:
                                break
                            data.append(json.loads(flow))
                        except:
                            print("Line {} has invalid character. Skipped ...".format(i))

                    max_flow_index = 0
                    str2txt = ""
                    for i, flow in enumerate(data):
                        for feature, value in flow.items():
                            # if TLS False and tls found then continue
                            if not TLS and feature.find('tls') > -1:
                                continue
                            # if DNS False and tls found then continue
                            if not DNS and feature.find('dns') > -1:
                                continue
                            # if HTTP False and tls found then continue
                            if not HTTP and feature.find('http') > -1:
                                continue
                            
                            if feature not in featureDict.keys():
                                featureDict[feature] = value
    
    fname = "featureDict_META"
    if TLS:
        fname += "_TLS"
    if DNS:
        fname += "_DNS"
    if HTTP:
        fname += "_HTTP"
    fname += ".txt"
    print("Done!\nWriting {} ...".format(fname))                        
    with open(fname, 'w') as file:
        str2txt += "{\n"
        for feature, value in featureDict.items():
            str2txt = str2txt + "\"" + feature + "\" : "
            if type(value) is list:
                str2txt += "["
                for i in range(len(value)-1):
                    str2txt = str2txt + str(i) + ", "
                str2txt = str2txt + str(len(value)-1) + "],\n"
            else:
                str2txt += "-1,\n"
        file.write(str2txt[:-2]+"\n}") # delete last new line, and comma, then put a new line and close curly brackets
    print("Done!")


def write2csv(datasetFolderName, data, label, feature_names, encrypted):
    if encrypted:
        filename = "/dataset_enc.csv"
    else:
        filename = "/dataset.csv"
    with open(datasetFolderName+filename, "w") as fwrite:
        for f in feature_names:
            fwrite.write(f)
            fwrite.write(",")
        fwrite.write("label\n")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                fwrite.write(str(data[i,j]))
                fwrite.write(",")
            fwrite.write(str(label[i,0]))
            fwrite.write("\n")


def plot_feature_importance(directory, featureList, importances, indices, nTop=None, title='None'):
    if nTop is None:
        nTop = 0
    plt.figure()
    plt.title(title)
    plt.barh(range(len(indices[-nTop:])), importances[indices[-nTop:]], color='b', align='center')
    plt.yticks(range(len(indices[-nTop:])), [featureList[i] for i in indices[-nTop:]])
    plt.xlabel('Relative Importance')
    if title.lower().find('corr') > 0 and title.lower().find('before') > 0:
        plt.savefig(directory+'/Top'+str(len(indices[-nTop:]))+'_b4rt-corr.png', bbox_inches='tight')
    elif title.lower().find('corr') > 0 and title.lower().find('before') < 0:
        plt.savefig(directory+'/Top'+str(len(indices[-nTop:]))+'_corr.png', bbox_inches='tight')
    elif title.lower().find('corr') < 0 and title.lower().find('before') > 0:
        plt.savefig(directory+'/Top'+str(len(indices[-nTop:]))+'_b4rt-imp.png', bbox_inches='tight')
    else:
        plt.savefig(directory+'/Top'+str(len(indices[-nTop:]))+'_imp.png', bbox_inches='tight')
 

def plot_cov_matrix(directory, dataframe, target=None):
    if target is None:
        corr = dataframe.corr()
    else:
        dataframe['label'] = target.values
        corr = dataframe.corr()
        
    plt.figure(figsize=(20,10))
    if dataframe.values.shape[1] > 60:
           sns.heatmap(abs(corr), annot=False, square=True, cmap='ocean')
    elif dataframe.values.shape[1] > 40:
        sns.heatmap(abs(corr), annot=False, square=True, cmap='ocean')
    else:
        sns.heatmap(abs(corr), annot=True, fmt='.1g', square=True, cmap='ocean')
    plt.title("Correlation Matrix")
    plt.savefig(directory+'/Top'+str(len(dataframe.columns)-1)+'_CorrMatrix.png', bbox_inches='tight')
    # Recover the dataframe
    dataframe.pop('label')
    

def plot_feature_correlation(directory, dataframe, target=None):
    if target is None:
        corr = dataframe.corr()
    else:
        dataframe['label'] = target.values
        corr = dataframe.corr()    
    
    cov_mat = abs(corr.values)
    correlation = cov_mat[-1,:-1] # exclude label-label correlation (1.0)
    featureList = dataframe.columns
    indices = np.argsort(correlation)
    plot_feature_importance(directory, featureList, correlation, indices, title='Feature Correlation')
    # Recover the dataframe
    dataframe.pop('label')

    return correlation


def plot_confusion_matrix(directory, y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    if n_classes == 2:
        detectionRate = cm[1,1]/(cm[1,0]+cm[1,1])
        falseAlarmRate = cm[0,1]/(cm[0,0]+cm[0,1])
        print("TPR: \t\t\t{:.5f}".format(detectionRate))
        print("FAR: \t\t\t{:.5f}".format(falseAlarmRate))
        if not title:
            if normalize:
                title = 'Normalized confusion matrix\nTPR:{:5f} - FAR:{:.5f}'.format(detectionRate, falseAlarmRate)
            else:
                title = 'Confusion matrix, without normalization\nTPR:{:.5f} - FAR:{:.5f}'.format(detectionRate, falseAlarmRate)
    else:
        F1_ = metrics.f1_score(y_true, y_pred, average="macro")
        #for c in range(cm.shape[0]):
        #    print(classes[c], metrics.average_precision_score(one_hot(y_true, n_classes)[:,c], one_hot(y_pred, n_classes)[:,c], average="weighted"))
        mAP = np.mean(np.asarray([(metrics.average_precision_score(one_hot(y_true, n_classes)[:,c], one_hot(y_pred, n_classes)[:,c], average="macro")) for c in range(cm.shape[0])]))
        print("F1: \t\t\t{:.5f}".format(F1_))
        print("mAP: \t\t\t{:.5f}".format(mAP))
        if not title:
            if normalize:
                title = 'Normalized confusion matrix\nF1:{:5f} - mAP:{:.5f}'.format(F1_, mAP)
            else:
                title = 'Confusion matrix, without normalization\nF1:{:.5f} - mAP:{:.5f}'.format(F1_, mAP)
    

    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], interpolation='nearest', cmap=cmap)
    #ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if n_classes < 4: # larger numbers cause too many digits on the confusion matrix
        fnt = 16
    elif n_classes < 8:
        fnt = 10
    else:
        fnt = max(4, 16-n_classes)

    fmt = '.2f' if normalize else 'd'
    thresh = np.sum(cm, axis=1) * 0.66
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i,j] != 0:
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        fontsize = fnt,
                        color="white" if cm[i, j] > thresh[i] else "black")

    fig.tight_layout()
    fig.savefig(directory+"/CM.png", bbox_inches='tight')
    print("Confusion matrix is saved as .{}/CM.png\n".format(directory[directory.find("/results"):]))

    return ax, cm


def retrain(directory, df, target, nTop, featureList, importances, indices, class_names, selection='FSMJ'):
    plt.close('all')
    # Performance measurements [t_train, acc, t_pred, Nmiss, Nfalse]
    perf = []
    #
    try:
        target = df.pop('label')
    except:
        pass

    to_drop = [featureList[i] for i in indices[:-nTop]]
    df = df.drop(to_drop, axis=1)

    # Plot covariance matrix and feature correlation with reduced features
    plot_cov_matrix(directory, df, target)
    correlation = plot_feature_correlation(directory, df, target)
    
    #
    Xtrain, Xtest, ytrain, ytest = train_test_split(df.values, target.values, test_size=0.2, random_state=42, shuffle = True, stratify = target)

    #
    scaler = preprocessing.StandardScaler()
    Xtrain_scaled = scaler.fit_transform(Xtrain)
    Xtest_scaled = scaler.transform(Xtest)

    #
    #clf = KNeighborsClassifier(n_neighbors=3)
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs = -1, max_features="auto")
    #clf = svm.SVC(C=1.0, kernel='rbf', random_state=10)
    #clf = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(int(2*len(list(df.columns))),), random_state=10)
    #clf = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(174,), random_state=10)
    
    # train 10 times and get average
    train_times = []
    for i in range(1):
        t0 = t.time()
        clf.fit(Xtrain_scaled, ytrain)
        t1 = t.time()
        train_times.append(t1-t0)
    perf.append(sum(train_times)/len(train_times))
    train_times = []
    
    #
    if selection == 'RF':
        # RF based importances / indices  
        feature_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs = -1, max_features="auto")
        feature_rf.fit(Xtrain_scaled, ytrain)
        importances = feature_rf.feature_importances_  
        indices = np.argsort(importances)
        featureList = list(df.columns)
    else:
        indices = np.argsort(importances)
        # Get non-nan number
        nonNan = np.sum(~np.isnan(importances))
        # Move nan to the end of array
        indices = np.roll(indices, len(indices)-nonNan)
        indices = indices[-nTop:]
        #importances = importances[indices]
        #indices = np.argsort(importances)

    plot_feature_importance(directory, featureList, importances, indices, nTop, title='Feature Importances-'+selection)
    
    #
    perf.append(clf.score(Xtest_scaled, ytest))
    print("Test Score: {:.4f}".format(clf.score(Xtest_scaled, ytest)))
    
    #
    # predict 10 times and get average
    pred_times = []
    for i in range(1):
        t0 = t.time()
        y_pred = clf.predict(Xtest_scaled)
        t1 = t.time()
        pred_times.append(t1-t0)
    perf.append(sum(pred_times)/len(pred_times))
    y_test = ytest

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    _, cm = plot_confusion_matrix(directory, y_test, y_pred, classes=class_names)

    # Plot normalized confusion matrix
    #plot_confusion_matrix(directory, y_test, y_pred, classes=class_names, normalize=True,
    #                      title='Normalized confusion matrix')
    # add miss
    perf.append(cm[1,0]/(cm[1,0]+cm[1,1]))
    # add false alarm
    perf.append(cm[0,1]/(cm[0,0]+cm[0,1]))

    plt.close('all')
        
    return perf
