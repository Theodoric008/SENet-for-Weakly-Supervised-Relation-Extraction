from IPython import embed
import pickle as pkl


mode_name = "WN"
pkl_file = "noNA_{}_aug_data_dict_entity_pair2aug_sentences.pkl".format(mode_name)
wn_dict = pkl.load(open(pkl_file, "rb"))

embed()

