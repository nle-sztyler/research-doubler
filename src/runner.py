import time
import pickle
import numpy as np
import os
import random as rn
from main import MAIN


def _load_pickle(file_path: str):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        return data


def _read_file(path_to_file):
    return np.loadtxt(path_to_file, delimiter=',', dtype='str')


def _testing(main_obj, test_data, test_neg_data):
    start_time = time.time()
    result_filename = str(int(start_time)) + '.txt'
    file = open(result_filename, 'w')

    test_disease_dict = dict()
    test_neg_disease_dict = dict()
    genes_neg = set(test_neg_data[:, 0])
    for gene, relation, disease in test_data:
        if disease not in test_disease_dict:
            test_disease_dict[disease] = set()
        test_disease_dict[disease].add(gene)
    for gene, relation, disease in test_neg_data:
        if disease not in test_neg_disease_dict:
            test_neg_disease_dict[disease] = set()
        test_neg_disease_dict[disease].add(gene)
    for disease, values in test_disease_dict.items():
        print('> Process: ' + str(disease))
        file.write('> Process: ' + str(disease) + '\n')
        print('>> ' + str(len(values)) + ' | ' + str(len(genes_neg)))
        file.write('>> ' + str(len(values)) + ' | ' + str(len(genes_neg)) + '\n')

        sample = np.array([[list(values)[0], 'gene_associated_with_disease', disease]])
        ranks, scores, nodes = main_obj.predict(sample)

        result = list(zip(nodes, scores[1]))
        result.sort(key=lambda x: x[1], reverse=True)
        result_filtered = list()
        for entry in result:
            if entry[0] in genes_neg or entry[0] in values:
                result_filtered.append(entry)

        # Recall-at-k
        recall_at_100 = 0
        for idx in range(100):
            gene = result_filtered[idx][0]
            if gene in values:
                recall_at_100 += 1
        value = recall_at_100 / len(values)
        print('>> Recall-at-100 = ' + str(value))
        file.write('Recall-at-100 = ' + str(value) + '\n')

    file.flush()
    file.close()

    return result_filename


def _initialize(gpu="0", seed=8675309, allow_growth=True):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    rn.seed(seed)
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'

    import tensorflow as tf

    # Limit operation to 1 thread for deterministic results.
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
    )
    session_conf.gpu_options.allow_growth = allow_growth

    from keras import backend as k

    tf.set_random_seed(seed)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    k.set_session(sess)


def main():
    # Init
    _initialize()

    # Data
    train_data = _read_file('../data/train.txt')
    valid_data = _read_file('../data/valid_gene_associated_with_disease.txt')
    valid_neg_data = _read_file('../data/valid_neg_gene_associated_with_disease.txt')
    test_data = _read_file('../data/test_gene_associated_with_disease.txt')
    test_neg_data = _read_file('../data/test_neg_gene_associated_with_disease.txt')
    bow_feature_path = '../data/features_bow.pkl'
    vocabulary_path = '../data/vocabulary.pkl'
    bow_feature, vocabulary = _load_pickle(bow_feature_path), _load_pickle(vocabulary_path)

    # create Instance
    obj = MAIN(batch_size=1024, epochs=10, num_negative=100, validation_step=10, normalize_score=True,
               bow_feature=bow_feature, vocabulary=vocabulary)

    # training and testing
    result = obj.fit_and_test(train_data, valid_data, valid_neg_data, _testing, test_data, test_neg_data)
    print(result)


if __name__ == '__main__':
    main()
