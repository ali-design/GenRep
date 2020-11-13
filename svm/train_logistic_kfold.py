# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
SVM training using 3-fold cross-validation.

Relevant transfer tasks: Image Classification VOC07 and COCO2014.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import ipdb
import argparse
import logging
import numpy as np
import os
import pickle
import sys
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import svm_helper

# create the logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def train_svm(opts):
    assert os.path.exists(opts.data_file), "Data file not found. Abort!"
    if not os.path.exists(opts.output_path):
        os.makedirs(opts.output_path)

    file_train = np.load(os.path.join(opts.data_file, 'train.npz'))
    features = file_train['features']
    targets = file_train['labels']
    # ipdb.set_trace()
    # normalize the features: N x 9216 (example shape)
    features = svm_helper.normalize_features(features)

    # parse the cost values for training the SVM on
    costs_list = svm_helper.parse_cost_list(opts.costs_list)
    logger.info('Training Classifier for costs: {}'.format(costs_list))

    # classes for which SVM training should be done
    if opts.cls_list:
        cls_list = [int(cls) for cls in opts.cls_list.split(",")]
    else:
        num_classes = targets.shape[1]
        cls_list = range(num_classes)
    logger.info('Training Classifier for classes: {}'.format(cls_list))

    for cost_idx in range(len(costs_list)):
        cost = costs_list[cost_idx]
        out_file, ap_out_file = svm_helper.get_logre_train_output_files(
            cost, opts.output_path
        )
        if os.path.exists(out_file) and os.path.exists(ap_out_file):
            logger.info('SVM model exists: {}'.format(out_file))
            logger.info('AP file exists: {}'.format(ap_out_file))
        else:
            logger.info('Training model with the cost: {}'.format(cost))
            clf = LogisticRegression(
                C=cost, intercept_scaling=1.0,
                verbose=0, solver='lbfgs',
                dual=False, max_iter=100,
            )
            cls_labels = np.argmax(targets, 1).astype(dtype=np.int32, copy=True)
            # meaning of labels in VOC/COCO original loaded target files:
            # label 0 = not present, set it to -1 as svm train target
            # label 1 = present. Make the svm train target labels as -1, 1.
            ap_scores = cross_val_score(
                clf, features, cls_labels, cv=3 
            )
            clf.fit(features, cls_labels)
            logger.info('cost: {} Accuracy: {} mean:{}'.format(
                 cost, ap_scores, ap_scores.mean()))
            logger.info('Saving cls cost AP to: {}'.format(ap_out_file))
            np.save(ap_out_file, np.array([ap_scores.mean()]))
            logger.info('Saving SVM model to: {}'.format(out_file))
            with open(out_file, 'wb') as fwrite:
                pickle.dump(clf, fwrite)


def main():
    parser = argparse.ArgumentParser(description='SVM model training')
    parser.add_argument('--data_file', type=str, default=None,
                        help="Numpy file containing image features")
    parser.add_argument('--targets_data_file', type=str, default=None,
                        help="Numpy file containing image labels")
    parser.add_argument('--output_path', type=str, default=None,
                        help="path where to save the trained SVM models")
    parser.add_argument('--costs_list', type=str, default="0.01,0.1",
                        help="comma separated string containing list of costs")
    parser.add_argument('--random_seed', type=int, default=100,
                        help="random seed for SVM classifier training")

    parser.add_argument('--cls_list', type=str, default=None,
                        help="comma separated string list of classes to train")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    opts = parser.parse_args()
    logger.info(opts)
    train_svm(opts)


if __name__ == '__main__':
    main()
