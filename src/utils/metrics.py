from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from . import cfg


class BatchMeter(object):
    """Calculate average/sum value after each time
    """
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.value = 0
        self.count = 0
    
    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
        
    def get_value(self, summary_type=None):
        if summary_type == 'mean':
            return self.avg
        elif summary_type == 'sum':
            return self.sum
        else:
           return self.value


def map_char2id():
    dict_char2id = {}
    dict_id2char = {}
    with open(cfg['Global']['character_dict_path'],'r', encoding='utf-8') as f_dict:
        char_list = f_dict.readlines()
        for i, char in enumerate(char_list):
            char = char.strip('\n')
            if char not in dict_char2id:
                dict_char2id[char] = i + 1
                dict_id2char[i + 1] = char
    f_dict.close()
    return dict_char2id, dict_id2char


def compute_acc(preds, labels, labels_len, id2char):
    correct_num = 0
    all_num = 0
    new_labels = []
    i = 0
    for char_len in labels_len:
        new_labels.append(labels[i: i+char_len])
        i += char_len
    for (pred), (target) in zip(preds, new_labels):
        pred = ''.join([id2char[int(c)] for c in pred])
        target = ''.join([id2char[int(c)] for c in target])
        pred = pred.replace(" ", "")
        target = target.replace(" ", "")
        if pred == target:
            correct_num += 1
        all_num += 1
    correct_num += correct_num
    all_num += all_num

    return correct_num/all_num