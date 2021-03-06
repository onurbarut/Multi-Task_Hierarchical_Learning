import os
import argparse
import time as t
import pandas as pd

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from utils.helper2 import *

def submit(clf, test_set, TLS, DNS, HTTP, scaler, class_label_pair, filepath):
    Xtest, ids = get_submission_data(test_set, TLS=TLS, DNS=DNS, HTTP=HTTP)
    X_test_scaled = scaler.transform(Xtest)
    print("Predicting on {} ...".format(test_set.split('/')[-1]))
    predictions = clf.predict(X_test_scaled)
    make_submission(predictions, ids, class_label_pair, filepath)   

def main ():
    parser = argparse.ArgumentParser(description="NetML Challenge 2020 Random Forest Classifier Baseline", add_help=True)
    parser.add_argument('-d', '--dataset', action="store", help="Dataset path")
    parser.add_argument('-a', '--anno', action="store", help="Annoation level: {top, mid, fine}")
    parser.add_argument('-s', '--submit', action="store", help="{test-std, test-challenge, both} Select which set to submit")
    parser.add_argument('-m', '--modelname', action="store", help="{RF, SVM, kNN, MLP} Select which model to train")

    args = parser.parse_args()

    if args.dataset == None or not os.path.isdir(args.dataset) or args.anno == None:
        print ("No valid dataset set or annotations found!")
        return
    elif args.submit is not None and args.submit not in ["test-std", "test-challenge", "both"]:
        print("Please select which set to submit: {test-std, test-challenge, both}")
        return
    elif args.anno not in ["top", "mid", "fine"]:
        print("Please select one of these for annotations: {top, mid, fine}. e.g. --anno top")
        return
    elif args.anno == "mid" and (args.dataset.find("NetML") > 0 or args.dataset.find("CICIDS2017") > 0):
        print("NetML and CICIDS2017 datasets cannot be trained with mid-level annotations. Please use either top or fine.")
        return
    else:
        training_set = args.dataset+"/2_training_set"
        training_anno_file = args.dataset+"/2_training_annotations/2_training_anno_"+args.anno+".json.gz"
        test_set = args.dataset+"/1_test-std_set"
        challenge_set = args.dataset+"/0_test-challenge_set"


    # Create folder for the results
    time_ = t.strftime("%Y%m%d-%H%M%S")
    time_ += '_{}_{}_{}'.format(args.dataset.split('/')[-1], args.modelname, args.anno)

    save_dir = os.getcwd() + '/results/' + time_
    os.makedirs(save_dir)

    # Specify if SMOTE is to be applied
    isSMOTE = False

    # TLS, DNS, HTTP features included?
    TLS , DNS, HTTP = {}, {}, {}
    TLS['tlsOnly'] = False # returns
    TLS['use'] = False
    TLS['n_common_client'] = 10
    TLS['n_common_server'] = 5
    #
    DNS['use'] = False
    ##
    ##
    #
    HTTP['use'] = False
    ##
    ##

    # Get training data in np.array format
    feature_names, Xtrain_ids, Xtrain, ytrain, class_label_pair = read_dataset(training_set, TLS=TLS, DNS=DNS, HTTP=HTTP, annotationFileName=training_anno_file, class_label_pairs=None)

    # Drop flows if either num_pkts_in or num_pkts_out < 1
    df = pd.DataFrame(data=Xtrain, columns=feature_names)
    df['label'] = ytrain
    isFiltered = df['num_pkts_in'] < 1
    f_df = df[~isFiltered]
    isFiltered = f_df['num_pkts_out'] < 1
    f_df = f_df[~isFiltered]
    target = f_df.pop('label')


    # Split validation set from training data
    Xtrain, Xval, ytrain, yval = train_test_split(f_df.values, target.values, 
                                                        test_size=0.2, 
                                                        random_state=10, 
                                                        shuffle = True, 
                                                        stratify = target)


    # Get name of each class to display in confusion matrix
    class_names = list(sorted(class_label_pair.keys()))

    # Preprocess the data
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(Xtrain)
    X_val_scaled = scaler.transform(Xval)

    # Train RF Model
    print("Training the model ...")
    clf = SVC(C=1.0, kernel='rbf', decision_function_shape='ovr', tol=1e-3, max_iter = 100000, random_state=42, verbose=True)
    clf.fit(X_train_scaled, ytrain)

    # Output accuracy of classifier
    print("Training Score: \t{:.5f}".format(clf.score(X_train_scaled, ytrain)))
    print("Validation Score: \t{:.5f}".format(clf.score(X_val_scaled, yval)))

    # Print Confusion Matrix
    ypred = clf.predict(X_val_scaled)

    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plot_confusion_matrix(directory=save_dir, y_true=yval, y_pred=ypred, 
                            classes=class_names, 
                            normalize=False)

    # Make submission with JSON format
    if args.submit == "test-std" or args.submit == "both":
        submit(clf, test_set, TLS, DNS, HTTP, scaler, class_label_pair, save_dir+"/submission_test-std.json")
    if args.submit == "test-challenge" or args.submit == "both":
        submit(clf, challenge_set, TLS, DNS, HTTP, scaler, class_label_pair, save_dir+"/submission_test-challenge.json")

if __name__ == "__main__":
    main()
