import tensorflow as tf
from tensorflow.keras import Model
from dataset import DataSet
import numpy as np
from tensorflow.keras.initializers import TruncatedNormal
from tqdm import tqdm
from time import time


class FISM(Model):

    def __init__(self, args):
        super(FISM, self).__init__()
        self.embedding_size = args['embedding_size']
        self.alpha = args['alpha']
        self.beta = args['beta']
        self.gamma = args['gamma']
        self.lambda_ = args['lambda_']
        self.verbose = args['verborse']
        self.num_items = args['num_items']
        self.num_users = args['num_users']
        self.confidence_factor = args['confidence_factor']
        self.Q_norms = None
        self.P_norms = None
        self.item_norms = None
        self.item_vectors = None
        self.P = tf.Variable(
            tf.random.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0, stddev=0.1))
        self.mask_value = tf.constant(0, shape=(1, self.embedding_size), dtype=tf.float32)
        self.Q = tf.Variable(
            tf.random.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0, stddev=0.1))
        self.bias_u = tf.keras.layers.Embedding(input_dim=self.num_users, output_dim=1,
                                                embeddings_initializer=TruncatedNormal(mean=0., stddev=0.1))
        self.bias_i = tf.keras.layers.Embedding(input_dim=self.num_items, output_dim=1,
                                                embeddings_initializer=TruncatedNormal(mean=0., stddev=0.1))

    def call(self, user_descriptions, user_ids, item_ids, num_items):
        user_bias = self.bias_u(user_ids)
        item_bias = self.bias_i(item_ids)
        P_with_mask = tf.concat([self.P, self.mask_value], axis=0)
        user_rated_items_embedding = tf.nn.embedding_lookup(P_with_mask, user_descriptions)
        items_embedding = tf.nn.embedding_lookup(self.Q, item_ids)
        user_des = tf.reduce_sum(user_rated_items_embedding, axis=1)
        coefficient = tf.pow(num_items, -tf.constant(self.alpha, dtype=tf.float32))
        r = tf.squeeze(user_bias) + tf.squeeze(item_bias) + tf.math.multiply(coefficient, tf.reduce_sum(
            tf.math.multiply(user_des, items_embedding), axis=1))
        return r

    def loss_fn_old(self, predictions, labels, ratings):
        confidences = 1 + self.confidence_factor * ratings
        loss = tf.reduce_sum(tf.math.multiply(confidences, tf.math.square(predictions - labels)))
        loss += self.beta * (tf.reduce_sum(tf.math.square(self.P)) + tf.reduce_sum(
            tf.math.square(self.Q)))
        loss += self.lambda_ * tf.reduce_sum(tf.math.square(self.bias_u.embeddings)) + self.gamma * tf.reduce_sum(
            tf.math.square(self.bias_i.embeddings))
        return loss

    def loss_fn(self, predictions, labels, ratings):
        predictions = tf.math.sigmoid(predictions)
        predictions = tf.clip_by_value(predictions, clip_value_min=1e-7, clip_value_max=1 - 1e-7)
        cross_entropy_elements = -(tf.math.multiply(labels, tf.math.log(predictions)) +
                                   tf.math.multiply(1 - labels, tf.math.log(1 - predictions)))
        confidences = 1 + self.confidence_factor * ratings
        loss = tf.reduce_sum(tf.math.multiply(confidences, cross_entropy_elements))
        loss += self.beta * (tf.reduce_sum(tf.math.square(self.P)) + tf.reduce_sum(tf.math.square(self.Q)))
        loss += self.lambda_ * tf.reduce_sum(tf.math.square(self.bias_u.embeddings)) + self.gamma * tf.reduce_sum(
            tf.math.square(self.bias_i.embeddings))
        return loss

    def prepare_for_prediction(self):
        self.Q_norms = tf.sqrt(tf.reduce_sum(tf.square(self.Q), axis=1))
        self.P_norms = tf.sqrt(tf.reduce_sum(tf.square(self.P), axis=1))
        self.item_vectors = tf.concat([self.P, self.Q], axis=1)
        self.item_norms = tf.sqrt(tf.reduce_sum(tf.square(self.item_vectors), axis=1))

    def sim_items(self, item_id, top_n: int = 100):
        item_embedded = tf.nn.embedding_lookup(self.P, item_id)
        item_embedded = tf.reshape(item_embedded, shape=(self.embedding_size, -1))
        scores = tf.matmul(self.Q, item_embedded)
        scores = tf.squeeze(scores)
        scores = scores / (self.Q_norms * self.P_norms[item_id])
        scores = scores.numpy()
        best = np.argpartition(scores, -top_n)[-top_n:]
        return sorted(zip(best, scores[best]), key=lambda x: -x[1])

    def sim_items_concat_pq(self, item_id, top_n: int = 100):
        item_embedded = tf.nn.embedding_lookup(self.item_vectors, item_id)
        item_embedded = tf.reshape(item_embedded, shape=(2 * self.embedding_size, -1))
        scores = tf.matmul(self.item_vectors, item_embedded)
        scores = tf.squeeze(scores)
        scores = scores / (self.item_norms * self.item_norms[item_id])
        scores = scores.numpy()
        best = np.argpartition(scores, -top_n)[-top_n:]
        return sorted(zip(best, scores[best]), key=lambda x: -x[1])


def predict_top_n(model, user_id, user_rated_items, top_n=100, batch_size=512):
    rated_items = set(user_rated_items[user_id])
    predicts = []
    user_descriptions = []
    user_ids = []
    item_ids = []
    num_items = []
    for item_id in range(model.num_items):
        if rated_items.__contains__(item_id):
            user_descriptions.append(list(rated_items.difference([item_id])) + [model.num_items])
            user_ids.append(user_id)
            item_ids.append(item_id)
            num_items.append(rated_items.__len__() - 1)
        else:
            user_descriptions.append(list(rated_items.difference([item_id])))
            user_ids.append(user_id)
            item_ids.append(item_id)
            num_items.append(rated_items.__len__())
        if user_descriptions.__len__() >= batch_size:
            batch_predict = model(np.array(user_descriptions, dtype=np.int32),
                                  np.array(user_ids, dtype=np.int32),
                                  np.array(item_ids, dtype=np.int32),
                                  np.array(num_items, dtype=np.float32))
            predicts += list(batch_predict.numpy())
            user_descriptions = []
            user_ids = []
            item_ids = []
            num_items = []
    batch_predict = model(np.array(user_descriptions, dtype=np.int32),
                          np.array(user_ids, dtype=np.int32),
                          np.array(item_ids, dtype=np.int32),
                          np.array(num_items, dtype=np.float32))
    predicts += list(batch_predict.numpy())
    items_score = [(iid, score) for iid, score in enumerate(predicts)]
    items_score.sort(key=lambda x: x[1], reverse=True)
    return items_score[:top_n]


def hit_rate_evaluate(fism_model, user_rated_items, dataset):
    total_items = 0
    in_train_count = 0
    count = 0
    count_hit = 0
    ndcg_users = []
    for user_id, rated_items in tqdm(dataset.test_users.items()):
        user_gains = []
        rec_top_n = predict_top_n(fism_model, user_id, user_rated_items, batch_size=256, top_n=10)
        top_item_ids = {rec_item[0] for rec_item in rec_top_n}
        for position, item_id in enumerate(rated_items):
            in_train_count += 1
            if top_item_ids.__contains__(item_id):
                count_hit += 1
                user_gains.append(1 / np.log(position + 2))
        idcg = 0
        for i in range(user_gains.__len__()):
            idcg += 1 / np.log(i + 2)
        if idcg > 0:
            ndcg_users.append(sum(user_gains) / idcg)
        total_items += rated_items.__len__()
        count += 1
        if count > 100:
            break
    in_train_rate = in_train_count / total_items
    hit_rate = count_hit / total_items
    ndcg = np.mean(ndcg_users)
    return in_train_rate, hit_rate, ndcg


def rank_score_evaluate(fism_model, user_rated_items, dataset):
    count = 0
    list_user_ranks = []
    num_item = dataset.num_items
    total_pred = 0
    pred_hit = 0
    for user_id, rated_items in tqdm(dataset.test_users.items()):
        list_rec_items = predict_top_n(fism_model, user_id, user_rated_items, batch_size=256, top_n=-1)
        rec_items_idx = {item_id: idx + 1 for idx, (item_id, score) in enumerate(list_rec_items)}
        user_ranks = []
        for item_id in rated_items:
            total_pred += 1
            if rec_items_idx.__contains__(item_id):
                pred_rank = rec_items_idx[item_id] / num_item
                user_ranks.append(pred_rank)
        list_user_ranks.append(user_ranks)
        count += 1
        if count > 100:
            break
    rank_mean_users = []
    for user_ranks in list_user_ranks:
        if user_ranks.__len__() > 0:
            rank_mean_users.append(np.mean(user_ranks))
    return np.mean(rank_mean_users), pred_hit / total_pred


@tf.function
def train_step(model, optimizer, user_descriptions, user_ids, item_ids, num_items, labels, ratings):
    with tf.GradientTape() as tape:
        predictions = model(user_descriptions, user_ids, item_ids, num_items)
        loss = model.loss_fn(predictions, labels, ratings)
    gradients = tape.gradient(target=loss, sources=model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def training(fism_model, optimizer, dataset, num_epochs, pretrained=False):
    epoch_step = tf.Variable(0, dtype=tf.int32)
    ckpt = tf.train.Checkpoint(fism_model=fism_model, epoch_step=epoch_step)
    manager = tf.train.CheckpointManager(checkpoint=ckpt, directory='./fism_ckpt', max_to_keep=3)
    if pretrained:
        ckpt.restore(manager.latest_checkpoint)
    user_rated_items = dataset.user_rated_items
    for epoch in range(num_epochs):
        train_loss = tf.constant(0, tf.float32)
        start_load_data = time()
        dataset.prepare_train_data()
        load_data_time = time() - start_load_data
        # print('done load data: ', load_data_time)
        start_train_time = time()
        for i in tqdm(range(dataset.num_batch)):
            user_descriptions, user_ids, item_ids, num_items, labels, ratings = dataset.get_batch(i)
            loss_step = train_step(fism_model, optimizer, user_descriptions, user_ids, item_ids, num_items, labels,
                                   ratings)
            train_loss += loss_step
        train_time = time() - start_train_time
        print('epoch: ', epoch, '. load data time: ', load_data_time, '. train time: ', train_time, '. train loss: ',
              train_loss.numpy() / (dataset.num_batch))
        if epoch % 2 == 0:
            fism_model.prepare_for_prediction()
            in_train_rate, user_hit_rate, ndcg = hit_rate_evaluate(fism_model, user_rated_items, dataset)
            user_rank_score, rank_in_train_set = rank_score_evaluate(fism_model, user_rated_items, dataset)

            score = {'ndcg': ndcg,
                     'cf_hit_rate': user_hit_rate,
                     'cf_in_train_set_rate': in_train_rate,
                     'cf_rank': user_rank_score}

            print('epoch: {}, score: {}'.format(epoch, score))
            ckpt.epoch_step.assign_add(epoch + 1)
            manager.save()
            print('done save at epoch: ', ckpt.epoch_step.numpy())


if __name__ == '__main__':
    base_folder = 'Data/'
    data = DataSet(base_folder + 'train.csv', base_folder + 'test.csv', negative_sample=1, batch_size=512)

    args = dict()
    args['embedding_size'] = 50
    args['alpha'] = 0.8
    args['beta'] = 0.0005
    args['gamma'] = 0.000
    args['lambda_'] = 0.000
    args['verborse'] = 1
    args['num_items'] = data.num_items
    args['num_users'] = data.num_users
    args['confidence_factor'] = 1

    fism = FISM(args)
    opt = tf.keras.optimizers.Adam(learning_rate=0.005)

    training(fism, opt, data, num_epochs=5)
