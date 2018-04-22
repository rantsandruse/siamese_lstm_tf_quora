import numpy as np
import random

# batch generator takes X1, X2 and y
# in this case y is already properly formatted:
# [ [1]
#   [2]
#   [3]
#   [4]
#   [5] ]

class BatchGenerator:
    def __init__(self, X_train1, X_train2, y_train, X_test1, X_test2, y_test, batchSize, shuffle=True):
        self.X_train1 = X_train1
        self.X_train2 = X_train2
        self.y_train = y_train

        self.X_test1 = X_test1
        self.X_test2 = X_test2
        self.y_test = y_test

        self.X_train_offset = 0
        self.X_test_offset = 0
        self.shuffle = shuffle
        self.batchSize = batchSize

    def shuffleIfTrue(self):
        if self.shuffle:
            arr = np.arange(len(self.X_train1))
            np.random.shuffle(arr)
            self.X_train1 = self.X_train1[arr]
            self.X_train2 = self.X_train2[arr]
            self.y_train = self.y_train[arr]


    def nextTrainBatch(self):
        start = self.X_train_offset
        end = self.X_train_offset + self.batchSize

        self.X_train_offset = end

        # handle wrap around    ,
        if end > len(self.X_train1):
            spillover = end - len(self.X_train1)
            self.X_train_offset = spillover
            X1 = np.concatenate((self.X_train1[start:], self.X_train1[:spillover]), axis = 0)
            X2 = np.concatenate((self.X_train2[start:], self.X_train2[:spillover]), axis=0)
            y = np.concatenate((self.y_train[start:], self.y_train[:spillover]), axis=0)
            self.X_train_offset = 0
            self.shuffleIfTrue()

        else:
            X1 = self.X_train1[start:end]
            X2 = self.X_train2[start:end]
            y = self.y_train[start:end]

        X1 = X1.astype(np.int32, copy = False)
        X2 = X2.astype(np.int32, copy = False)
        y = y.astype(np.float32, copy = False)

        return X1,X2,y

    def nextTestBatch(self):
        start = self.X_test_offset
        end = self.X_test_offset + self.batchSize
        self.X_test_offset = end

        # handle wrap around    ,
        if end > len(self.X_test1):
            spillover = end - len(self.X_test1)
            self.X_test_offset = spillover
            X1 = np.concatenate((self.X_test1[start:], self.X_test1[:spillover]), axis=0)
            X2 = np.concatenate((self.X_test1[start:], self.X_test2[:spillover]), axis=0)
            y = np.concatenate((self.y_test[start:], self.y_test[:spillover]), axis=0)

            self.X_test_offset = 0
            self.shuffleIfTrue()

        else:
            X1 = self.X_test1[start:end]
            X2 = self.X_test2[start:end]
            y = self.y_test[start:end]

        X1 = X1.astype(np.int32, copy=False)
        X2 = X2.astype(np.int32, copy=False)
        y = y.astype(np.float32, copy=False)

        return X1, X2, y



