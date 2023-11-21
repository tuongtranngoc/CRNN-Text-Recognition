from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np


def post_processing(labels, blank=0):
    mapped_labels = []
    prev_label = None

    for l in labels:
        if l != prev_label:
            mapped_labels.append(l)
            prev_label = l
    mapped_labels = [l for l in mapped_labels if l != blank]
    return mapped_labels


def best_path_decode(log_probs):
    labels_list = []
    log_probs = log_probs.transpose((1, 0, 2))
    for log_prob in log_probs:
        labels = np.argmax(log_prob, axis=-1)
        labels = post_processing(labels)
        labels_list.append(labels)
    return labels_list


def prefix_search_decode():
    pass

def beam_search_withLLM():
    pass