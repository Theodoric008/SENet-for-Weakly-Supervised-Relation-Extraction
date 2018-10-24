# -*- coding:utf-8 -*

import random
import os
import numpy as np
import copy
import nltk
from nltk.tag import map_tag
from nltk.corpus import wordnet as wn
from nltk.tag import StanfordNERTagger
import kenlm
from IPython import embed
from tqdm import tqdm
from nltk.tree import Tree
from collections import defaultdict
import pickle as pkl
import concurrent.futures

# Tag		Meaning					English Examples
# ADJ		adjective				new, good, high, special, big, local
# ADP		adposition				on, of, at, with, by, into, under
# ADV		adverb					really, already, still, early, now
# CONJ		conjunction				and, or, but, if, while, although
# DET		determiner, article		the, a, some, most, every, no, which
# NOUN		noun					year, home, costs, time, Africa
# NUM		numeral					twenty-four, fourth, 1991, 14:24
# PRT		particle				at, on, out, over per, that, up, with
# PRON		pronoun					he, their, her, its, my, I, us
# VERB		verb					is, say, told, given, playing, would
# .			punctuation marks		. , ; !
# X			other					ersatz, esprit, dunno, gr8, university
def tag_filter(tag):
    if tag == "CONJ":
        return False
    elif tag == "ADP":
        return False
    elif tag == "DET":
        return False
    elif tag == "NUM":
        return False
    elif tag == "PRT":
        return False
    elif tag == ".":
        return False
    elif tag == "X":
        return False
    return True


def tag_wn(tag):
    if tag == 'n':
        return 'NOUN'
    elif tag == 'v':
        return 'VERB'
    elif tag == 'a' or tag == 's':
        return 'ADJ'
    elif tag == 'r':
        return 'ADV'
    else:
        return 'NONE'


lm = kenlm.Model('./kenlm_model/en-70k-0.2.lm')


def sample_by_lm(rep, post_s, index):
    word = post_s[index][0]
    a = max(index - 2, 0)
    b = min(index + 3, len(post_s))
    li = []
    for i in range(a, b):
        li.append(post_s[i][0])
    score = []
    candi = []

    for w in rep:
        li[index - a] = w
        t2 = " ".join(li)
        s2 = lm.score(t2)
        score.append(s2)
        candi.append(w)

    score = np.array(score)
    score = np.exp(score / 2 - 5)
    t_score = np.sum(score)
    score = score / t_score
    s = np.random.random_sample()
    i = 0
    while s - score[i] > 0:
        s -= score[i]
        i += 1
    return candi[i]


# WN
def random_wordnet_noise(words, vocab, prob=0.0):
	sentence = copy.deepcopy(words)
	if random.random() < prob:
		ner_tag = st.tag(sentence)
		post = nltk.pos_tag(sentence)
		index = 0
		aug_sentence = []
		for (x, t) in post:
			t = map_tag('en-ptb', 'universal', t)
			synlist = [x]
			if tag_filter(t) and ner_tag[index][1] == 'O':
				for syn in wn.synsets(x):
					sss = syn.name().split('.')[0]
					ttt = syn.name().split('.')[1]
					if vocab.get(sss, 0) != 0 and t == tag_wn(ttt):
						synlist += [str(sss)]
				synlist = list(set(synlist))
			# embed()
			if np.random.random_sample() <= 0.9:
				temp = sample_by_lm(synlist, post, index)
				aug_sentence.append(temp)
			# print(temp)
			else:
				temp = x
				aug_sentence.append(temp)
			index += 1
		sentence = aug_sentence
	sentence = " ".join(sentence)
	return sentence


st = StanfordNERTagger('./stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz',
                       './stanford-ner-2018-10-16/stanford-ner.jar')


# ************* cfit start ***************
cfit_list = list(open("CFit/overfitting.dict").readlines())
cfit_list = [s.strip() for s in cfit_list]
cfit_dict = dict()

for i in range(1, len(cfit_list), 2):
	s = cfit_list[i]
	t = cfit_list[i + 1].split("\t")
	cfit_dict[s] = t[0:5]


def random_cfit_noise(words, prob=0.0):
	sentence = copy.deepcopy(words)
	if random.random() < prob:
		ner_tag = st.tag(sentence)
		index = 0
		aug_sentence = []
		for k in sentence:
			if cfit_dict.get(k, 0) != 0 and ner_tag[index][1] == 'O':
				if np.random.random_sample() <= 0.9:
					aug_sentence.append(sample_by_lm(cfit_dict[k], sentence, index))
				else:
					aug_sentence.append(k)
			else:
				aug_sentence.append(k)
			index += 1
		sentence = aug_sentence
	sentence = " ".join(sentence)
	return sentence

# ************* cfit end ***************

# ************* comp start ***************
comp_C = defaultdict(dict)


def load_model():
	text = list(open("./Comp/written.model").readlines())
	text = [s.strip() for s in text]

	for s in text:
		comp_l = s.split(" ")
		s1 = comp_l[0]
		s2 = comp_l[1]
		c = float(comp_l[2]) / float(comp_l[3])
		comp_C[s1][s2] = comp_C.get(s1, {}).get(s2, c)


def decoder(r, f, k, aug_sentence):
	if type(r) != Tree:
		# print(r)
		aug_sentence.append(r)
		return

	thresh = comp_C.get(f, {}).get(r.label(), -1)
	p = np.random.random_sample()
	# print p, thresh
	if thresh != -1 and p < thresh:
		return

	for i in range(0, len(r)):
		decoder(r[i], r.label(), k + 1, aug_sentence)


# dep_parser = st
from nltk.parse.stanford import StanfordParser
dep_parser = StanfordParser(path_to_jar="./stanford-parser-full-2018-10-17/stanford-parser.jar",
                            path_to_models_jar="./stanford-parser-full-2018-10-17/stanford-parser-3.9.2-models.jar")

load_model()


def random_comp_noise(words, prob=0.0):
    sentence = copy.deepcopy(words)
    if random.random() < prob:
        sentence = " ".join(sentence)
        comp_a = list(dep_parser.raw_parse(sentence))
        aug_sentence = []
        decoder(comp_a[0], "root", 0, aug_sentence)
        sentence = aug_sentence
    sentence = " ".join(sentence)
    return sentence


# ************* comp end ***************


# gen start
orig_file_name = "train.txt"
# 组装关系集合
relations = ['/sports/sports_team/location', '/business/shopping_center/owner', '/people/family/members',
             '/people/person/place_of_birth', '/location/us_state/capital', '/broadcast/producer/location',
             '/people/person/children', '/time/event/locations', '/location/jp_prefecture/capital',
             '/people/person/place_lived', '/location/country/languages_spoken', '/people/person/nationality',
             '/location/administrative_division/country', '/people/deceased_person/place_of_death',
             '/people/person/religion', '/business/company/locations', '/location/in_state/judicial_capital',
             '/film/film_location/featured_in_films', '/location/mx_state/capital',
             '/location/in_state/legislative_capital', '/people/ethnicity/included_in_group',
             '/business/shopping_center_owner/shopping_centers_owned', '/business/business_location/parent_company',
             '/location/province/capital', '/film/film_festival/location', '/location/country/capital',
             '/business/company/major_shareholders', '/location/br_state/capital',
             '/people/profession/people_with_this_profession', '/people/deceased_person/place_of_burial',
             '/location/neighborhood/neighborhood_of', '/people/person/ethnicity', '/location/cn_province/capital',
             '/people/person/profession', '/business/company/place_founded', '/film/film/featured_film_locations',
             '/location/us_county/county_seat', '/people/family/country', '/business/company/advisors',
             '/business/company_advisor/companies_advised', '/location/fr_region/capital',
             '/location/de_state/capital', '/base/locations/countries/states_provinces_within',
             '/people/place_of_interment/interred_here', 'NA', '/people/ethnicity/geographic_distribution',
             '/location/in_state/administrative_capital', '/business/company/founders', '/broadcast/content/location',
             '/location/country/administrative_divisions', '/location/location/contains', '/location/it_region/capital',
             '/business/person/company']


# 组装词汇字典
vocab_list = pkl.load(open("vocab_list.pkl", "rb"))
vocab_dict = {}
for vocab in vocab_list:
    vocab_dict[vocab] = 1


# 读取训练数据
training_data = list(open(orig_file_name, encoding='utf-8').readlines())
training_data = [s.split() for s in training_data]


def process_data_item(data_item):
    entity1 = data_item[2]
    entity2 = data_item[3]
    if data_item[4] not in relations:
        relation = "NA"
    else:
        relation = data_item[4]
    if relation == "NA":
        return None
    else:
        sentence = data_item[5: -1]
        aug_sentences_list = []
        for loop in tqdm(range(0, 10)):
            aug_sentence = random_wordnet_noise(sentence, vocab_dict, 1.0)
            # aug_sentence = random_comp_noise(sentence, 1.0)
            # aug_sentence = random_cfit_noise(sentence, 1.0)
            aug_sentences_list.append(aug_sentence)
        aug_sentences = list(set(aug_sentences_list))
        entity_pair = (entity1, entity2)
        return entity_pair, sentence, aug_sentences


# 并行处理
with concurrent.futures.ProcessPoolExecutor() as executor:
    res_dict = {}
    res_ex_it = executor.map(process_data_item, tqdm(training_data[0:100]))
    print("map is done, encapsulation start")
    res_ex_it = list(res_ex_it)
    print("to list is done, scanning...")
    for res_ex in res_ex_it:
        if not res_ex:
            continue
        entity_pair = res_ex[0]
        sentence = res_ex[1]
        sentence = " ".join(sentence)
        aug_sentences = res_ex[2]
        if entity_pair not in res_dict.keys():
            res_dict[entity_pair] = {sentence: aug_sentences}
        else:
            res_dict[entity_pair][sentence] = aug_sentences

    mode_name = "WN100"
    os.system("rm -f noNA_{}_aug_data_dict_entity_pair2aug_sentences.pkl".format(mode_name))
    pkl.dump(res_dict, open("noNA_{}_aug_data_dict_entity_pair2aug_sentences.pkl".format(mode_name), "wb"))


# # 顺序处理
# res_dict = {}
# for i in tqdm(training_data[0:100]):
#     entity_pair, sentence, aug_sentences = process_data_item(i)
#     sentence = " ".join(sentence)
#     if entity_pair not in res_dict.keys():
#         res_dict[entity_pair] = {sentence: aug_sentences}
#     else:
#         res_dict[entity_pair][sentence] = aug_sentences
# mode_name = "WN"
# os.system("rm -f noNA_{}_aug_data_dict_entity_pair2aug_sentences.pkl".format(mode_name))
# pkl.dump(res_dict, open("noNA_{}_aug_data_dict_entity_pair2aug_sentences.pkl".format(mode_name), "wb"))
