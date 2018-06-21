import keras.backend as K
import tensorflow as tf

import numpy as np


class Performance():

    def __init__(self, y_true, y_pred, y_max=0):

        self.y_pred = y_pred
        self.y_true = y_true

        self.classes = y_max + 1

        self.trues_and_falses()
        self.f1score()
        self.balance()

    def f1score(self):

        '''Computes fscore when possible'''

        if self.ses(self.pos_pred) == self.length:
            if self.ses(self.pos) != self.length:
                self.result = '_warning_all_ones_'
                return
        elif self.ses(self.pos_pred) == 0:
            if self.ses(self.pos) != 0:
                self.result = '_warning_all_zeros_'
            elif self.ses(self.pos) == 0:
                self.result = 1

        try:
            self.precision = tf.divide(self.tp, tf.add(self.tp, self.fp))
            tf.divide(1, self.precision)
        except ZeroDivisionError:
            self.result = '_warning_no_true_positive'
            return

        try:
            self.recall = tf.divide(self.tp, tf.add(self.tp, self.fn))
            tf.divide(1, self.recall)
        except ZeroDivisionError:
            self.result = '_warning_3'
            return

        try:
            f1_prod = tf.multiply(self.precision, self.recall)
            f1_sum = tf.add(self.precision, self.recall)
            f1_div = tf.divide(f1_prod, f1_sum)
            self.result = tf.multiply(np.float64(2), f1_div)
        except ZeroDivisionError:
            return

    def trues_and_falses(self):

        self.length = self.y_true.get_shape().as_list()[0]
        categories = self.y_true.get_shape().as_list()[1]
        self.ses = tf.Session().run

        self.tp = tf.constant(0)
        self.tn = tf.constant(0)
        self.fp = tf.constant(0)
        self.fn = tf.constant(0)

        self.pos = K.sum(self.y_true)
        self.neg = tf.constant(self.length) - self.pos
        self.pos_pred = K.sum(self.y_pred)
        self.neg_pred = tf.constant(self.length) - self.pos_pred

        for i in range(categories):
            for j in range(self.length):

                pred = self.y_pred[j][i]
                true = self.y_true[j][i]

                if self.ses(pred) == 1 and self.ses(true) == 1:
                    self.tp = tf.add(self.tp, 1)
                elif self.ses(pred) == 1 and self.ses(true) == 0:
                    self.fn = tf.add(self.fn, 1)
                elif self.ses(pred) == 0 and self.ses(true) == 0:
                    self.tn = tf.add(self.tn, 1)
                elif self.ses(pred) == 0 and self.ses(true) == 1:
                    self.fp = tf.add(self.fn, 1)

    def balance(self):

        self.balance = tf.divide(self.pos, tf.constant(self.length))

    def zero_rule(self):

        return tf.add(-self.balance, 1)

    def one_rule(self):

        return self.balance

    def get_result(self):

        return self.result
