import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

#load in data
new_data_small=np.loadtxt(open("new_data_small.csv","r"),dtype=str,delimiter=",")
new_data_small = new_data_small[1:-1]
np.random.shuffle(new_data_small)
new_data_small_test = new_data_small[100000:120000]
new_data_small = new_data_small[0:100000]

#data preprocessing, drop the label and the month feature
x_train = np.concatenate((new_data_small[:,1:2],new_data_small[:,3:-1]),axis=1)
y_train = new_data_small[:,2:3]
y_train_input=np.array(list(map(int,np.squeeze(y_train))))
x_test = np.concatenate((new_data_small_test[:,1:2],new_data_small_test[:,3:-1]),axis=1)
y_test_input = np.array(list(map(int,np.squeeze(new_data_small_test[:,2:3]))))

#simplified version, picking only day of week, depature block, carrier name, previous airports and depature airport as features to encode
def poor_version():
    x_train_discrete = x_train[:, [0, 2, 6, 15, 18]]
    onehot = OneHotEncoder(handle_unknown='ignore')
    onehot.fit(x_train_discrete)
    pickle.dump(onehot, open('enc.pkl', 'wb'))
    x_train_input = onehot.transform(x_train_discrete).toarray()
    x_test_discrete = x_test[:, [0, 2, 6, 15, 18]]
    x_test_input = onehot.transform(x_test_discrete).toarray()

    return x_train_input,x_test_input

x_train_input, x_test_input = poor_version()

# cross_validation
# input: x -- training data features
#        y -- training data labels
#        n_fold -- number of subsets
#        model -- model instance
# output : mean of accuracy, error rate, precision, recall of the cross validation
def cross_validation(x,y,n_fold,model):
    cv_y_accuracy = np.array([])
    n=x.shape[0] # data size
    n_val=n//n_fold # subset size
    for i in range(n_fold):
        train_index = []
        # calculate the validation set range
        validation_index = list(range(i * n_val, (i + 1) * n_val))
        # calculate the training set range
        if i > 0:
            train_index = list(range(i * n_val))
        if i < n - 1:
            train_index = train_index + list(range((i + 1) * n_val, n))
        # transform lists to numpy arrays to ease calculation
        train_index = np.array(train_index)
        validation_index = np.array(validation_index)
        # calculate y_predict
        cv_y_predict = model.fit(x[train_index.astype(np.int32), :], y[train_index.astype(np.int32)]).predict(
            x[validation_index, :])
        # calculate y_test
        cv_y_test = y[validation_index.astype(np.int32)]

        cv_y_accuracy = np.append(cv_y_accuracy, (np.sum(cv_y_predict == cv_y_test)) / (cv_y_test.shape[0]))
    return cv_y_accuracy


# Fit model and use pickle to save the model to be used in the backend server
decision_tree = DecisionTreeClassifier(max_depth=4, random_state=0)
decision_tree.fit(x_train_input,y_train_input)
pickle.dump(decision_tree, open('model.pkl','wb'))
