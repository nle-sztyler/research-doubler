import collections
import numpy as np
import tqdm
import utils
from keras.layers import Input
from keras.layers.core import Activation
from keras.layers.merge import Average
from keras.models import Model
from keras import losses
from keras import optimizers
import keras.backend as K
from numpy.random import RandomState
from doubler import Doubler


InputParamTuple = collections.namedtuple('InputParamTuple', 'num_negative batch_size epochs validation_step'
                                                            ' loss optimizer activation_fn normalize')
Data = collections.namedtuple('Data', 'triples_train triples_train_idx triples_validation')
Graph = collections.namedtuple('Graph', 'vertices vertex_indexer num_vertices index_vertex relations'
                                        ' relation_indexer index_relation num_relations')


class MAIN:
    def __init__(self, batch_size: int = 256, epochs: int = 100, activation_fn: str = 'softmax',
                 num_negative: int = 500, optimizer: optimizers = optimizers.Adam(),
                 loss: losses = 'categorical_crossentropy', validation_step: int = 1, normalize_score: bool = False,
                 bow_feature=None, vocabulary=None):

        self._model_params = InputParamTuple(num_negative=num_negative, batch_size=batch_size, epochs=epochs,
                                             validation_step=validation_step,
                                             optimizer=optimizer,
                                             loss=loss,
                                             activation_fn=activation_fn,
                                             normalize=normalize_score)  # TODO

        self._data = None
        self._model = None
        self._best_model = None
        self._normalize_score = normalize_score

        self._doubler = Doubler(embedding_node_dim=300, embedding_doc_dim=64,
                                bow_feature=bow_feature, vocabulary=vocabulary)

    def fit_and_test(self, triples_train: np.array, triples_validation: np.array, triples_validation_neg: np.array,
                     test_func, test_data, test_neg_data) -> list:
        self.__load_graph(triples_train, triples_validation)
        self.__build_model()
        return self.__train_and_test(triples_validation_neg, test_func, test_data, test_neg_data)

    def predict(self, triples_test: np.array):
        known_triples = utils.get_known_triples(self._data, self._gp)
        triples = utils.extract_triples(triples_test, self._gp.vertex_indexer, self._gp.relation_indexer)
        known_triples = known_triples.union(triples)

        nodes_idx = []
        relations_idx = []

        for triple in triples_test:
            nodes_idx.append(self._gp.vertex_indexer[triple[0]])
            nodes_idx.append(self._gp.vertex_indexer[triple[2]])
            relations_idx.append(self._gp.relation_indexer[triple[1]])
            relations_idx.append(self._gp.relation_indexer[triple[1]])

        ranks, scores = self.__predict_nodes(nodes_idx, relations_idx, known_triples)

        return ranks, scores, self._gp.vertices

    def __load_graph(self, triples_train: np.array, triples_validation: np.array):
        # build graph
        self.__build_graph(triples_train)

        # convert training triples to training_idx triples
        triples_train_idx = [[self._gp.vertex_indexer[triple[0]],
                              self._gp.relation_indexer[triple[1]],
                              self._gp.vertex_indexer[triple[2]]]
                             for triple in triples_train]

        # build data structure
        self._data = Data(triples_train=triples_train,
                          triples_train_idx=triples_train_idx,
                          triples_validation=triples_validation)

    def __build_graph(self, triples_train):
        nodes = list(set(triples_train[:, 0]).union(triples_train[:, 2]))
        nodes.sort()
        node_indexer = {node: idx for idx, node in enumerate(nodes)}

        relations = list(set(triples_train[:, 1]))
        relations.sort()
        relation_indexer = {rel: idx for idx, rel in enumerate(relations)}

        num_nodes = len(node_indexer)
        num_relations = len(relation_indexer)
        index_node = {idx: vertex for vertex, idx in node_indexer.items()}
        index_relation = {idx: relation for relation, idx in relation_indexer.items()}

        self._gp = Graph(vertices=nodes,
                         vertex_indexer=node_indexer,
                         num_vertices=num_nodes,
                         index_vertex=index_node,
                         relations=relations,
                         relation_indexer=relation_indexer,
                         index_relation=index_relation,
                         num_relations=num_relations)

    def __build_model(self):
        input_layer_positive = Input(shape=(1,), name='input_node_positive')
        input_layer_negative = Input(shape=(self._model_params.num_negative + 1,), name='input_node_negative')
        input_layer_relation = Input(shape=(1,), name='input_relation')

        # keep track of all input and output layers
        input_layer_list = [input_layer_positive, input_layer_negative, input_layer_relation]
        output_layer_list = []

        # build
        input_layers, score_layer, l2_offset_layer = self._doubler.build_model(self._model_params, self._gp,
                                                                               input_layer_positive,
                                                                               input_layer_negative,
                                                                               input_layer_relation)
        output_layer_list.append(score_layer)
        input_layer_list.extend(input_layers)

        # combine score layers
        if len(output_layer_list) > 1:
            score_layer_total = Average()(output_layer_list)
        else:
            score_layer_total = output_layer_list[0]

        # build model
        score_layer_total = Activation(self._model_params.activation_fn, name='output_overall_score')(score_layer_total)

        if l2_offset_layer is not None:
            l2_offset_layer = Activation('relu', name='output_overall_offset')(l2_offset_layer)
            self._model = Model(inputs=input_layer_list, outputs=[score_layer_total, l2_offset_layer])
            self._model.compile(loss=self.__custom_loss, optimizer=self._model_params.optimizer)
        else:
            self._model = Model(inputs=input_layer_list, outputs=[score_layer_total])
            self._model.compile(loss=self._model_params.loss, optimizer=self._model_params.optimizer)

        # print model
        print(self._model.summary())

    def __custom_loss(self, y_true, y_pred):
        if 'offset' in y_pred.name:
            result = K.sum(K.abs(y_true - y_pred))/1000000.0
            return K.minimum(result, 10)
        else:
            return K.categorical_crossentropy(y_true, y_pred)

    def __train_and_test(self, triples_validation_neg, test_func, test_data, test_neg_data) -> list:
        result_files = list()

        generator_token = self.__generator()
        steps_per_epoch = int(len(self._data.triples_train_idx) / self._model_params.batch_size)

        for epoch in range(self._model_params.epochs):
            print('Epoch %s/%s' % ((epoch + 1), self._model_params.epochs))

            self._model.fit_generator(generator_token, steps_per_epoch, epochs=1, shuffle=False,
                                      workers=1, use_multiprocessing=False)

            if (epoch + 1) % self._model_params.validation_step == 0:
                self._validation(self._data.triples_validation, triples_validation_neg)

            result_file = test_func(self, test_data, test_neg_data)
            result_files.append(result_file)

        return result_files

    def _validation(self, valid_data, valid_neg_data):
        val_disease_dict = dict()
        val_neg_disease_dict = dict()
        recall_scores = list()
        genes_neg = set(valid_neg_data[:, 0])
        for gene, relation, disease in valid_data:
            if disease not in val_disease_dict:
                val_disease_dict[disease] = set()
            val_disease_dict[disease].add(gene)
        for gene, relation, disease in valid_neg_data:
            if disease not in val_neg_disease_dict:
                val_neg_disease_dict[disease] = set()
            val_neg_disease_dict[disease].add(gene)
        for disease, values in tqdm.tqdm(val_disease_dict.items(), total=len(val_disease_dict.keys()), desc='> Validation'):
            sample = np.array([[list(values)[0], 'gene_associated_with_disease', disease]])
            ranks, scores, nodes = self.predict(sample)

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
            recall_score = recall_at_100 / len(values)
            recall_scores.append(recall_score)

        print('\nRecall-AT-100 (Mean) = ' + str(np.mean(recall_scores)) +
              '| Recall-AT-100 (Std) = ' + str(np.std(recall_scores)))

    def __predict_nodes(self, nodes_idx: list, relations_idx: list, known_triples: set):
        scores = np.zeros((len(nodes_idx), self._gp.num_vertices))

        doubler_score_node, doubler_score_doc = self._doubler.predict_nodes(nodes_idx, relations_idx)
        if self._normalize_score:
            doubler_score_node = utils.sigmoid(doubler_score_node)
            if doubler_score_doc is not None:
                doubler_score_doc = utils.sigmoid(doubler_score_doc)
        scores = np.sum([scores, doubler_score_node], axis=0)
        if doubler_score_doc is not None:
            scores = np.sum([scores, doubler_score_doc], axis=0)

        ranks = []
        num_scores = len(scores)
        for idx, row in tqdm.tqdm(enumerate(scores), total=num_scores, desc='> Compute Ranking'):
            is_head = idx % 2 == 0
            node_idx_given = nodes_idx[idx]
            relation_idx_given = relations_idx[idx]
            node_idx_wanted = nodes_idx[idx + 1] if is_head else nodes_idx[idx - 1]
            threshold_lower = row[node_idx_wanted]

            rank_cleaned = 1
            for node_idx, score in enumerate(row):
                if score <= threshold_lower:
                    continue
                elif is_head and not (node_idx_given, relation_idx_given, node_idx) in known_triples:
                    rank_cleaned += 1
                elif not is_head and not (node_idx, relation_idx_given, node_idx_given) in known_triples:
                    rank_cleaned += 1

            ranks.append(rank_cleaned)

        return ranks, scores

    def __generator(self):
        steps_per_epoch = int(len(self._data.triples_train_idx) / self._model_params.batch_size)

        while True:
            for idx in range(steps_per_epoch):
                input_dict = self.__generate_data(idx, steps_per_epoch)

                # build batch
                batch = []
                for input_layer in self._model.inputs:
                    layer_name = input_layer.name
                    layer_name = layer_name[:layer_name.rfind(':')]
                    batch.append(input_dict[layer_name])

                # perfect result
                y1 = self.__generate_output_data(input_dict['input_relation'].shape[0])
                y2 = np.zeros(len(y1))
                y = [y1, y2]

                yield batch, y

    def __generate_output_data(self, num_triples: int):
        y = [0] * (self._model_params.num_negative + 1)
        y[0] = 2  # because we have 1+1 (node+doc)
        y = np.array([y] * num_triples)

        return y

    def __generate_data(self, batch_idx: int, num_batches: int):
        node_train_idx_positive = []
        node_train_idx_negatives = []
        relation_train_idx = []
        generated_training_data = {}
        batch_size = self._model_params.batch_size
        num_negative = self._model_params.num_negative

        if batch_idx == num_batches - 1:
            triples_train_idx_batch = self._data.triples_train_idx[batch_idx * batch_size:]
        else:
            triples_train_idx_batch = self._data.triples_train_idx[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        self._doubler.init_batch_triples(self._model_params, batch_idx, num_batches, triples_train_idx_batch)

        np_random = RandomState(batch_idx)
        random_indices = np_random.randint(self._gp.num_vertices, size=(2 * len(triples_train_idx_batch), num_negative))

        for idx, triple_idx in enumerate(triples_train_idx_batch):
            relation_train_idx.append(triple_idx[1])
            relation_train_idx.append(triple_idx[1])
            node_train_idx_positive.append([triple_idx[2]])
            node_train_idx_positive.append([triple_idx[0]])

            replacement = int((triple_idx[0] + triple_idx[2]) / 2)

            neg_head_replacement = replacement - 1 if replacement > 0 else replacement
            head_idx_negatives = random_indices[idx, ]
            head_idx_negatives = np.where(head_idx_negatives == triple_idx[0], neg_head_replacement, head_idx_negatives)
            head_idx_negatives = np.insert(head_idx_negatives, 0, triple_idx[0])
            node_train_idx_negatives.append(head_idx_negatives)

            neg_tail_replacement = replacement + 1 if replacement < (self._gp.num_vertices - 1) else replacement
            tail_idx_negatives = random_indices[batch_size - 1 + idx, ]
            tail_idx_negatives = np.where(tail_idx_negatives == triple_idx[2], neg_tail_replacement, tail_idx_negatives)
            tail_idx_negatives = np.insert(tail_idx_negatives, 0, triple_idx[2])
            node_train_idx_negatives.append(tail_idx_negatives)

        training_data = self._doubler.generate_training_data(node_train_idx_positive,
                                                             node_train_idx_negatives)
        generated_training_data.update(training_data)

        # store default input data
        generated_training_data.update({'input_node_positive': np.array(node_train_idx_positive)})
        generated_training_data.update({'input_node_negative': np.array(node_train_idx_negatives)})
        generated_training_data.update({'input_relation': np.array(relation_train_idx)})

        return generated_training_data
