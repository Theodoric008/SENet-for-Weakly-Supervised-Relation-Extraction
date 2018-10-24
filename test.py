# coding=utf-8

import numpy as np
import pickle as pkl
from tqdm import tqdm


def test(testing_data, input_x, input_p1, input_p2, s, p, dropout_keep_prob, datamanager, sess, num_epoch):
    results = []
    i = 0
    for test in tqdm(testing_data):
        i += 1
        x_test = datamanager.generate_x(testing_data[test])
        p1, p2 = datamanager.generate_p(testing_data[test])
        scores, pre = sess.run([s, p], {input_x: x_test, input_p1: p1, input_p2: p2, dropout_keep_prob: 1.0})
        max_pro = 0
        prediction = -1
        score_ = None
        for score in scores:
            score = np.exp(score-np.max(score))
            score = score/score.sum(axis=0)
            score[0] = 0
            score_ = score[1:]
            pro = score[np.argmax(score)]
            if pro > max_pro and np.argmax(score) != 0:
                max_pro = pro
                prediction = np.argmax(score)
        results.append((test, testing_data[test][0].relation.id, max_pro, prediction, score_))
    outpacks = {"result": results}
    print("writing test result of num_epoch: {} ...".format(num_epoch))
    pkl.dump(outpacks, open("temp/precision_recall_{}.pkl".format(num_epoch), 'wb'))
