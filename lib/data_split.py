from sklearn.cross_validation import StratifiedShuffleSplit


def train_test_split_shuffle(target, features1, features2, test_size = 0.2):
    #print(target.shape)
    #print(features1.shape)
    #print(features2.shape)
    sss = StratifiedShuffleSplit(target, 1, test_size = test_size, random_state=59)
    for train_index, test_index in sss:
        X_train1, X_test1 = features1[train_index], features1[test_index]
        X_train2, X_test2 = features2[train_index], features2[test_index]
        y_train, y_test = target[train_index], target[test_index]


        #y_test = y_test.values
        #y_train = y_train.values

    return X_train1, X_train2, y_train, X_test1, X_test2, y_test


