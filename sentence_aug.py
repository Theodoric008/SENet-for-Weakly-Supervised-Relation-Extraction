# -*- coding:utf-8 -*

import random
import copy
from IPython import embed


# WD
def random_word_dropout(e1, e2, words, prob=0.0):
    sentence = copy.deepcopy(words)
    if random.random() < prob:
        rand_pos = random.randint(0, len(words) - 1)
        while words[rand_pos] == e1 or words[rand_pos] == e2:
            rand_pos = random.randint(0, len(words) - 1)
        sentence[rand_pos] = ""
    return sentence


# flip
def random_sentence_flip(words, prob=0.0):
    sentence = copy.deepcopy(words)
    if random.random() < prob:
        sentence = sentence[::-1]
    return sentence


# linguistic adversity
def random_lin_adv_noise(words, e1, e2, pkl_dict, prob=0.0):
    sentence = copy.deepcopy(words)
    if random.random() < prob:
        entity_pair = (e1, e2)
        if entity_pair not in pkl_dict.keys():
            return sentence
        if " ".join(sentence) not in pkl_dict[entity_pair].keys():
            return sentence
        aug_sentences = pkl_dict[entity_pair][sentence]
        chosen_sentence = random.sample(aug_sentences, 1)[0]
        sentence = chosen_sentence.split()
    return sentence
