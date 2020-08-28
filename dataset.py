import numpy as np
import pandas as pd
from tqdm import tqdm


class DataSet:

    def __init__(self, train_file, test_file, negative_sample=3, batch_size=64):
        self.train_data = pd.read_csv(train_file)
        self.test_data = pd.read_csv(test_file)
        self.num_users = self.train_data['user_id'].max() + 1
        self.num_items = self.train_data['item_id'].max() + 1
        self.negative_sample = negative_sample
        self.batch_size = batch_size
        self.user_rated_items = self.get_user_rated_items()
        self.num_batch = -1
        self.all_train_data = None
        self.test_users = self.get_test_user()

    def get_test_user(self):
        test_user_dict = dict()
        for user_id, item_id in self.test_data[['user_id', 'item_id']].values:
            if not test_user_dict.__contains__(user_id):
                test_user_dict[user_id] = []
            test_user_dict[user_id].append(item_id)
        return test_user_dict

    def get_user_rated_items(self):
        rated_data = self.get_rated_data()
        user_rated_items = dict()
        for user_id, item_id, rate in rated_data:
            if not user_rated_items.__contains__(user_id):
                user_rated_items[user_id] = set()
            user_rated_items[user_id].add(item_id)
        return user_rated_items

    def prepare_train_data(self):
        rated_data = self.get_rated_data()
        np.random.shuffle(rated_data)
        self.all_train_data = self.negative_sampling(rated_data)
        self.num_batch = self.all_train_data[0].__len__()//self.batch_size

    def get_batch(self, i):
        user_ids, item_ids, labels, ratings = self.all_train_data
        batch_user_descriptions = []
        batch_item_ids = item_ids[i * self.batch_size: (i + 1) * self.batch_size]
        batch_user_ids = user_ids[i * self.batch_size: (i + 1) * self.batch_size]
        batch_num_items = []
        batch_labels = labels[i * self.batch_size: (i + 1) * self.batch_size]
        batch_ratings = ratings[i * self.batch_size: (i + 1) * self.batch_size]
        mask = self.num_items
        for j in range(self.batch_size):
            idx = i * self.batch_size + j
            user_id = user_ids[idx]
            item_id = item_ids[idx]
            rated_items = self.user_rated_items[user_id].copy()
            user_description = self.get_user_description(rated_items, item_id)
            batch_user_descriptions.append(user_description)
            batch_num_items.append(user_description.__len__())
        max_user_des = max(batch_num_items)
        batch_user_descriptions = self.padding_user_description(batch_user_descriptions, mask, max_user_des)
        return (batch_user_descriptions,
                np.array(batch_user_ids, dtype=np.int32),
                np.array(batch_item_ids, dtype=np.int32),
                np.array(batch_num_items, dtype=np.float32),
                np.array(batch_labels, dtype=np.float32),
                np.array(batch_ratings, dtype=np.float32)
                )

    def generate_train_data(self):
        rated_data = self.get_rated_data()
        np.random.shuffle(rated_data)
        all_train_data = self.negative_sampling(rated_data)
        all_batch_data = self.get_all_batch_data(all_train_data, self.user_rated_items)
        return all_batch_data

    def get_rated_data(self):
        return [(user_id, item_id, rate) for user_id, item_id, rate in self.train_data[['user_id', 'item_id', 'rating']].values]

    def negative_sampling(self, rated_data):
        user_ids = []
        item_ids = []
        labels = []
        ratings = []
        set_rated = {(user_id, item_id) for user_id, item_id, rating in rated_data}
        for user_id, item_id, rating in rated_data:
            user_ids.append(user_id)
            item_ids.append(item_id)
            labels.append(1)
            ratings.append(rating)
            for j in range(self.negative_sample):
                random_item = np.random.randint(self.num_items)
                while set_rated.__contains__((user_id, random_item)):
                    random_item = np.random.randint(self.num_items)
                user_ids.append(user_id)
                item_ids.append(random_item)
                labels.append(0)
                ratings.append(0)
        return user_ids, item_ids, labels, ratings

    def get_user_description(self, rated_items, item_id):
        if rated_items.__contains__(item_id):
            rated_items.remove(item_id)
        return list(rated_items)

    def padding_user_description(self, batch_user_descriptions, mask, max_len):
        result = np.zeros([len(batch_user_descriptions), max_len], dtype=np.int32) + mask
        for idx, user_des in enumerate(batch_user_descriptions):
            result[idx][0:len(user_des)] = user_des
        return result


if __name__ == '__main__':
    # base_folder = 'F:\\Projects\\Train\\Python\\recommendation-tensorflow-2\\Data\\'
    base_folder = 'Data/'
    dataset = DataSet(base_folder + 'train.csv', base_folder + 'test.csv', batch_size=512, negative_sample=1)
    dataset.prepare_train_data()
    # all_batch = dataset.generate_train_data()
    for i in tqdm(range(dataset.num_batch)):
        user_descriptions, user_ids, item_ids, num_items, labels, ratings = dataset.get_batch(i)
