import numpy as np
import keras

from keras.layers import Embedding
from keras.layers.core import Reshape
from keras.layers.merge import Multiply, Dot, Add
from keras.layers import Input, Dense

import custom_layers
from collections import namedtuple


class Doubler:
    def __init__(self, embedding_node_dim: int, embedding_doc_dim: int, bow_feature: dict, vocabulary: set):
        self._name = 'doubler_'
        self._batch_negative = None
        self._node_embedding = None
        self._relation_embedding_node = None
        self._relation_embedding_doc = None
        self._doc_embedding = None
        self._graph_properties = None
        self._embedding_node_dim = embedding_node_dim
        self._embedding_doc_dim = embedding_doc_dim
        self._bow_feature = {str(key): value for key, value in bow_feature.items()}
        self._vocabulary = vocabulary

    def build_model(self, model_params: namedtuple, graph_properties: namedtuple,
                    input_layer_positive: keras.layers.Input,
                    input_layer_negative: keras.layers.Input,
                    input_layer_relation: keras.layers.Input):

        self._graph_properties = graph_properties

        # Input Documents
        input_layer_doc_positive = Input(shape=(len(self._vocabulary),), name='input_document_positive')
        input_layer_doc_negative = Input(shape=(model_params.num_negative + 1, len(self._vocabulary),),
                                         name='input_document_negative')

        # node, relation, and document embeddings
        self._node_embedding = Embedding(graph_properties.num_vertices, self._embedding_node_dim,
                                         name=self._name + 'node_embedding')
        self._doc_embedding = Dense(self._embedding_doc_dim, name=self._name + 'document_embedding')
        self._relation_embedding_node = Embedding(graph_properties.num_relations, self._embedding_node_dim,
                                                  name=self._name + 'relation_embedding')

        if self._embedding_node_dim == self._embedding_doc_dim:
            self._relation_embedding_doc = self._relation_embedding_node
        else:
            self._relation_embedding_doc = Embedding(graph_properties.num_relations, self._embedding_doc_dim,
                                                     name=self._name + 'relation_embedding_2')

        # connect node, relation, and doc input and corresponding embedding layer
        embedding_node_layer_positive = self._node_embedding(input_layer_positive)
        embedding_node_layer_negative = self._node_embedding(input_layer_negative)
        embedding_doc_layer_positive = self._doc_embedding(input_layer_doc_positive)
        embedding_doc_layer_negative = self._doc_embedding(input_layer_doc_negative)
        embedding_layer_relation_node = self._relation_embedding_node(input_layer_relation)

        # we have to create a separate relation embedding layer for the documents in case that the
        # target dimension of the node and document embeddings differs
        embedding_layer_relation_doc = embedding_layer_relation_node if self._embedding_node_dim == self._embedding_doc_dim \
            else self._relation_embedding_doc(input_layer_relation)

        # compute the center of the negative node embeddings
        mean_layer_node = custom_layers.Mean(name=self._name + 'avg_node_layer_negative')
        avg_node_layer_negative = mean_layer_node(embedding_node_layer_negative)

        # compute the distance between the positive node and the avg. negative node embedding
        shape = embedding_node_layer_positive.get_shape().as_list()
        tmp = Reshape(((shape[1] * shape[2]),))(embedding_node_layer_positive)  # shape[1] is always 1
        diff_layer_node = custom_layers.L2Diff(name=self._name + 'node_L2_diff')
        node_dis = diff_layer_node([avg_node_layer_negative, tmp])

        # compute the center of the negative doc embeddings
        mean_layer_doc = custom_layers.Mean(name=self._name + 'avg_doc_layer_negative')
        avg_doc_layer_negative = mean_layer_doc(embedding_doc_layer_negative)

        # compute the distance between the positive doc and the avg. negative doc embedding
        diff_layer_doc = custom_layers.L2Diff(name=self._name + 'doc_L2_diff')
        doc_dis = diff_layer_doc([avg_doc_layer_negative, embedding_doc_layer_positive])

        # compute L2_Offset
        l2_offset_layer = custom_layers.L2Off(name=self._name + 'L2_offset')([node_dis, doc_dis])

        # create node score layer
        embedding_layer_node_joint = Multiply(name=self._name + 'embedding_layer_node_joint')(
            [embedding_layer_relation_node, embedding_node_layer_positive])
        output_layer_node_score = Dot(axes=2, name=self._name + 'node_score')(
            [embedding_layer_node_joint, embedding_node_layer_negative])
        output_layer_node_score = Reshape((model_params.num_negative + 1,))(output_layer_node_score)

        # create doc score layer
        embedding_layer_doc_joint = Multiply(name=self._name + 'embedding_layer_doc_joint')(
            [embedding_layer_relation_doc, embedding_doc_layer_positive])
        output_layer_doc_score = Dot(axes=2, name=self._name + 'document_score')(
            [embedding_layer_doc_joint, embedding_doc_layer_negative])
        output_layer_doc_score = Reshape((model_params.num_negative + 1,))(output_layer_doc_score)

        # create final score/predicate layer
        output_layer_score = Add(name=self._name + 'score')([output_layer_node_score, output_layer_doc_score])

        return [input_layer_doc_positive, input_layer_doc_negative], output_layer_score, l2_offset_layer

    def predict_node(self, node_idx: int, relation_idx: int):
        vertex_emb_matrix = self._node_embedding.get_weights()[0]
        relation_emb_node_matrix = self._relation_embedding_node.get_weights()[0]
        scores_emb_node = np.dot((vertex_emb_matrix[node_idx] * relation_emb_node_matrix[relation_idx]),
                                 vertex_emb_matrix.T)

        nodes_candidate = np.zeros((self._graph_properties.num_vertices, len(self._vocabulary)), dtype=np.int8)
        for node_idx_candidate, node_candidate in enumerate(self._graph_properties.vertices):
            tail_bow = self._bow_feature[node_candidate] if node_candidate in self._bow_feature else np.zeros(
                len(self._vocabulary))
            nodes_candidate[node_idx_candidate, :] = tail_bow

        doc_emb_matrix = np.dot(nodes_candidate, self._doc_embedding.get_weights()[0]) + self._doc_embedding.get_weights()[1]
        relation_emb_doc_matrix = self._relation_embedding_doc.get_weights()[0]
        scores_emb_doc = np.dot((doc_emb_matrix[node_idx] * relation_emb_doc_matrix[relation_idx]), doc_emb_matrix.T)

        return scores_emb_node, scores_emb_doc

    def predict_nodes(self, nodes_idx: list, relations_idx: list):
        scores_node = list()
        scores_doc = list()

        for idx, node_idx in enumerate(nodes_idx):
            score_node, score_doc = self.predict_node(node_idx, relations_idx[idx])
            scores_node.append(score_node)
            scores_doc.append(score_doc)

        return np.array(scores_node), np.array(scores_doc)

    def init_batch_triples(self, model_params: namedtuple, batch_idx: int, num_batches: int, triples_batch: list):
        batch_size = (2 * len(triples_batch)) if batch_idx == num_batches - 1 else 2 * model_params.batch_size
        self._batch_negative = np.zeros((batch_size, model_params.num_negative + 1, len(self._vocabulary)),
                                        dtype=np.int8)

    def generate_training_data(self, nodes_train_idx: list, nodes_train_idx_candidate: list):
        batch_positive = np.zeros((len(nodes_train_idx), len(self._vocabulary)), dtype=np.int8)

        for idx, node_train_idx in enumerate(nodes_train_idx):
            node_train = self._graph_properties.index_vertex[node_train_idx[0]]

            if node_train not in self._bow_feature:
                continue

            batch_positive[idx, ] = self._bow_feature[node_train]

            tmp_array = np.zeros((self._batch_negative.shape[1], self._batch_negative.shape[2]))
            nodes_train_idx_neg = nodes_train_idx_candidate[idx]
            for idx2, node_train_idx_neg in enumerate(nodes_train_idx_neg):
                node_train_neg = self._graph_properties.index_vertex[node_train_idx_neg]

                if node_train_neg not in self._bow_feature:
                    continue

                node_train_doc_neg = self._bow_feature[node_train_neg]
                tmp_array[idx2, ] = node_train_doc_neg

            self._batch_negative[idx, ] = tmp_array

        return {'input_document_positive': batch_positive,
                'input_document_negative': self._batch_negative}
