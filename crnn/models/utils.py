from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

def decode(text_index, text_prob=None, is_remove_duplicate=False):
    characters = "-0123456789abcdefghijklmnopqrstuvwxyz"
    results_list = []
    ignored_tokens = [0]
    batch_size = len(text_index)
    for batch_idx in range(batch_size):
        char_list = []
        conf_list = []
        for idx in range(len(text_index[batch_idx])):
            if text_index[batch_idx][idx] in ignored_tokens:
                continue
            if is_remove_duplicate:
                if idx > 0 and text_index[batch_idx][idx-1] == text_index[batch_idx][idx]:
                    continue
            char_list.append(characters[int(text_index[batch_idx][idx])])
            if text_prob is not None:
                conf_list.append(text_prob[batch_idx][idx])
            else:
                conf_list.append(1.0)
        text = ''.join(char_list)
        results_list.append((text, np.mean(conf_list)))
    return results_list