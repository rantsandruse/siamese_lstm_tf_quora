from batch_generator import BatchGenerator
import tensorflow as tf
from lib import features_word2vec
import data_split
import pandas as pd
import datetime
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#word2vecmodel, embedding_weights, word_features, data = data_prep()
# To do tomorrow:
# 1. rename params for the alpha part
# 2. find out

class SiameseLSTMModel:

    # initiate everything
    # embedding weights: map index to word vecs
    # word_features: map

    # oos: out of sample train/test split.
    def __init__(self, features1_train, features2_train, label_train,\
                 embedding_weights, maxSeqLength, \
                 restore = False
                 ):

        tf.reset_default_graph()
        self.session = tf.Session()
        self.restore = restore

        if restore:
           self.restore_model()

        self.initialize_params(maxSeqLength)
        self.initialize_filepaths()

        self.initialize_inputs(embedding_weights)

        self.initialize_train_test_split(features1_train, features2_train, label_train)

        self.initialize_model()
        self.initialize_tboard()


    # initialize all hyperparameters
    # including batch size, network size etc.
    def initialize_params(self, maxSeqLength):
        self.batchSize = 128
        self.lstmUnits = 64
        self.numClasses = 2
        self.maxSeqLength = maxSeqLength
        self.numDimensions = 300
        self.dropOutRate = 0.20
        self.margin = 1

        self.trainSize = 0

    def initialize_filepaths(self):
        self.lstm_model_path = "./model/pretrained_lstm_tf.model"


    # initialize inputs
    # word embeddings
    # I have moved the rest to
    def initialize_inputs(self, embedding_weights):
        # Read data
        # Use the kaggle Bag of words vs Bag of popcorn data:
        # https://www.kaggle.com/c/word2vec-nlp-tutorial/data
        self.embedding_weights = embedding_weights.astype("float32")


    # Split train and test
    # This is good for quora question pair
    def initialize_train_test_split(self, features1_train, features2_train, label_train):
        pairedlabel = np.array([[int(x)] for x in label_train])

        self.X_train1, self.X_train2, self.y_train, self.X_test1, self.X_test2, self.y_test = \
            data_split.train_test_split_shuffle(pairedlabel, features1_train, features2_train, test_size=0.1)
        self.myBatchGenerator = BatchGenerator(self.X_train1, self.X_train2, self.y_train, self.X_test1, self.X_test2, self.y_test, self.batchSize)

    # initialize model weights, placeholders etc.
    # And model cell itself
    def initialize_model(self):
        if self.restore:
            self.restore_old_model()
        else:
           self.create_new_model()

    # https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
    # http: // www.mit.edu / ~jonasm / info / MuellerThyagarajan_AAAI16.pdf
    # use exp(-|h1 - h2|) and rms loss
    def contrastive_loss_modified(self, pairedlabel, lstm_out1, lstm_out2, batchSize):
        with tf.name_scope("contrastive_loss_modified"):
            # use mahanttan distance here.
            distance = -tf.reduce_sum(tf.abs(tf.subtract(lstm_out1, lstm_out2)), 1, keep_dims=True)

            # how about cross entropy loss?
            # return tf.reduce_mean(
            #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.distance, labels=pairedlabel))

            return tf.losses.mean_squared_error(pairedlabel, tf.exp(distance)), distance

    def create_new_model(self):
        self.input_data1 = tf.placeholder(tf.int32, [self.batchSize, self.maxSeqLength], name = "input_data1" )
        self.input_data2 = tf.placeholder(tf.int32, [self.batchSize, self.maxSeqLength], name="input_data2")

        self.labels = tf.placeholder(tf.float32, [self.batchSize, 1], name="input_label")

        self.data1 = tf.Variable(tf.zeros([self.batchSize, self.maxSeqLength, self.numDimensions]), dtype=tf.float32)
        self.data1 = tf.nn.embedding_lookup(self.embedding_weights, self.input_data1, name = "data1")

        self.data2 = tf.Variable(tf.zeros([self.batchSize, self.maxSeqLength, self.numDimensions]), dtype=tf.float32)
        self.data2 = tf.nn.embedding_lookup(self.embedding_weights, self.input_data2, name = "data2")

        self.weight = tf.Variable(tf.random_normal([self.lstmUnits, 1], stddev=0.1), \
                               name = "weight")
        self.bias = tf.Variable(tf.random_normal([1], stddev=0.1), \
                             name = "bias" )

        lstmCell = tf.contrib.rnn.BasicLSTMCell(self.lstmUnits)
        self.lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=1 - self.dropOutRate)

        print("creating a new model:\n")
        # dimension: 128, 500, 64
        # print(self.data1.get_shape())
        self.output1, _ = tf.nn.dynamic_rnn(self.lstmCell, self.data1, dtype=tf.float32)
        # dimension, 500, 128, 64
        self.output1 = tf.transpose(self.output1, [1, 0, 2])
        # dimension: 128, 64
        self.last1 = tf.gather(self.output1, int(self.output1.get_shape()[0]) - 1)

        self.output2, _ = tf.nn.dynamic_rnn(self.lstmCell, self.data2, dtype=tf.float32)

        self.output2 = tf.transpose(self.output2, [1, 0, 2])
        self.last2 = tf.gather(self.output2, int(self.output2.get_shape()[0]) - 1)

        # what if we use cross entropy loss instead of
        #self.loss = tf.reduce_mean(
        #    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.distance, labels=self.labels))

        self.loss, self.distance = self.contrastive_loss_modified(self.labels, self.last1, self.last2, self.batchSize)

        self.prediction = tf.rint(tf.exp(self.distance), name="prediction")
        self.correctPred = tf.equal(self.prediction, self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(self.correctPred, tf.float32), name="accuracy")

        self.optimizer = tf.train.AdamOptimizer(0.001)
        gradients, variables = zip(*(self.optimizer.compute_gradients(self.loss)))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        self.optimize = self.optimizer.apply_gradients(zip(gradients, variables))

        tf.add_to_collection('output1', self.output1)
        tf.add_to_collection('output2', self.output2)
        tf.add_to_collection('last1', self.last1)
        tf.add_to_collection('last2', self.last2)
        tf.add_to_collection('prediction', self.prediction)
        tf.add_to_collection('correctPred', self.correctPred)
        tf.add_to_collection('accuracy', self.accuracy)
        tf.add_to_collection('loss', self.loss)
        tf.add_to_collection('optimizer', self.optimize)


    def restore_old_model(self, graph_file = './model/pretrained_lstm_tf.model-0.meta', checkpoint_dir= './model'  ):
        self.saver = tf.train.import_meta_graph(graph_file)
        self.saver.restore(self.session, tf.train.latest_checkpoint(checkpoint_dir))

        graph = tf.get_default_graph()

        self.weight = graph.get_tensor_by_name("weight:0")
        self.bias = graph.get_tensor_by_name("bias:0")

        self.input_data = graph.get_tensor_by_name('input_data:0')
        self.data = graph.get_tensor_by_name('data:0')
        self.output1 = tf.get_collection('output1')[0]
        self.output2 = tf.get_collection('output2')[0]

        self.last = tf.get_collection('last')[0]
        self.prediction = tf.get_collection('prediction')[0]
        self.correctPred = tf.get_collection('correctPred')[0]
        self.accuracy = tf.get_collection('accuracy')[0]
        self.loss = tf.get_collection('loss')[0]
        self.optimizer = tf.get_collection('optimizer')[0]


    # initialize tensor board for monitoring
    def initialize_tboard(self):
        tf.summary.scalar('Loss', self.loss)
        tf.summary.scalar('Accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()
        self.logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        self.writer = tf.summary.FileWriter(self.logdir, self.session.graph)
        if not self.restore:
            self.saver = tf.train.Saver()

    def train_single_epoch(self, epoch_num = 0):
        i = 0
        while True:
            # Next Batch of reviews
            nextBatch1, nextBatch2, nextBatchLabels = self.myBatchGenerator.nextTrainBatch()
            if len(nextBatch1) * (i+1) > len(self.X_train1): break

            self.session.run(self.optimize, {self.input_data1: nextBatch1, self.input_data2: nextBatch2, self.labels: nextBatchLabels})

            # Write summary to Tensorboard
            if (i % 10 == 0):
                summary, acc, cost = self.session.run([ self.merged, self.accuracy, self.loss], {self.input_data1: nextBatch1, self.input_data2: nextBatch2, self.labels: nextBatchLabels})
                print "Iter " + str(i) + ", Minibatch Loss= " + "{:.6f}".format(cost) + \
                      ", Training Accuracy= " + "{:.5f}".format(acc)

            i += 1

    def train_epochs(self, n_epochs):
        if not self.restore:
            tf.global_variables_initializer().run(session=self.session)

        num = 0
        while num < n_epochs:
            print("Epoch " + str(num) + ":\n")
            self.train_single_epoch(num)
            self.test()

            if num % 5 == 0:
                self.save_model(num)
                print("saved to %s" % self.lstm_model_path)

            num += 1

        self.writer.flush()
        self.writer.close()
        print('training finished.')

    def test(self):
        i = correct = total = 0

        while True:
            nextBatch1, nextBatch2, nextBatchLabels = self.myBatchGenerator.nextTestBatch()

            if len(nextBatch1) * (i + 1) > len(self.X_test1): break

            acc = self.session.run(self.accuracy, {self.input_data1: nextBatch1, self.input_data2: nextBatch2, self.labels: nextBatchLabels})
            correct += acc
            total += len(nextBatch1)

            i += 1

        total_accuracy = correct/i if i else 0
        print("Testing accuracy = " + "{:.5f}".format(total_accuracy))


    def save_model(self, step_num ):
        self.saver.save(self.session,
                     self.lstm_model_path, global_step=step_num)

