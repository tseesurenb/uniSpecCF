'''
Minimal Dataloader
'''

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import world
from os.path import join


class BasicDataset:
    """Base dataset class"""
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        return len(self.trainUser)
    
    @property
    def valDataSize(self):
        return getattr(self, '_valDataSize', 0)
    
    @property
    def testDict(self):
        return self._testDict
    
    @property
    def valDict(self):
        return getattr(self, '_valDict', {})
    
    @property
    def allPos(self):
        return self._allPos
    
    def getUserPosItems(self, users):
        """Get positive items for users"""
        posItems = []
        for user in users:
            user_idx = int(user)
            posItems.append(self.UserItemNet[user_idx].nonzero()[1])
        return posItems


class LastFM(BasicDataset):
    """LastFM dataset"""
    
    def __init__(self, path="../data/lastfm", val_ratio=0.1):
        print("Loading LastFM...")
        
        trainData = pd.read_table(join(path, 'data1.txt'), header=None) - 1
        testData = pd.read_table(join(path, 'test1.txt'), header=None) - 1
        
        self.testUser = np.array(testData[0])
        self.testItem = np.array(testData[1])
        
        self._create_train_val_split(trainData, val_ratio)
        
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_users, self.m_items)
        )
        
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self._testDict = self._build_dict(self.testUser, self.testItem)
        self._valDict = self._build_dict(self.valUser, self.valItem) if hasattr(self, 'valUser') else {}

    def _create_train_val_split(self, trainData, val_ratio):
        """Split training data"""
        user_items = {}
        for _, row in trainData.iterrows():
            user, item = row[0], row[1]
            if user not in user_items:
                user_items[user] = []
            user_items[user].append(item)
        
        train_users, train_items = [], []
        val_users, val_items = [], []
        
        np.random.seed(42)
        for user, items in user_items.items():
            if len(items) == 1:
                train_users.append(user)
                train_items.extend(items)
            else:
                n_val = max(1, int(len(items) * val_ratio))
                val_indices = np.random.choice(len(items), n_val, replace=False)
                
                for i, item in enumerate(items):
                    if i in val_indices:
                        val_users.append(user)
                        val_items.append(item)
                    else:
                        train_users.append(user)
                        train_items.append(item)
        
        self.trainUser = np.array(train_users)
        self.trainItem = np.array(train_items)
        self.valUser = np.array(val_users)
        self.valItem = np.array(val_items)
        self._valDataSize = len(val_users)

    def _build_dict(self, users, items):
        """Build user->items dictionary"""
        data_dict = {}
        for user, item in zip(users, items):
            if user not in data_dict:
                data_dict[user] = []
            data_dict[user].append(item)
        return data_dict

    @property
    def n_users(self):
        return 1892
    
    @property
    def m_items(self):
        return 4489


class ML100K(BasicDataset):
    """ML-100K dataset"""
    
    def __init__(self, path="../data/ml-100k", val_ratio=0.1):
        print("Loading ML-100K...")
        
        train_data = self._load_coo_file(join(path, 'train_coo.txt'))
        test_data = self._load_coo_file(join(path, 'test_coo.txt'))
        
        self.testUser = np.array([x[0] for x in test_data])
        self.testItem = np.array([x[1] for x in test_data])
        
        all_data = train_data + test_data
        self.n_user = max([x[0] for x in all_data]) + 1
        self.m_item = max([x[1] for x in all_data]) + 1
        
        self._create_train_val_split(train_data, val_ratio)
        
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item)
        )
        
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self._testDict = self._build_dict(self.testUser, self.testItem)
        self._valDict = self._build_dict(self.valUser, self.valItem) if hasattr(self, 'valUser') else {}

    def _load_coo_file(self, file_path):
        """Load COO format file"""
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            user_id = int(parts[0])
                            item_id = int(parts[1])
                            data.append((user_id, item_id))
                        except ValueError:
                            continue
        return data

    def _create_train_val_split(self, train_data, val_ratio):
        """Split training data"""
        user_items = {}
        for user, item in train_data:
            if user not in user_items:
                user_items[user] = []
            user_items[user].append(item)
        
        train_users, train_items = [], []
        val_users, val_items = [], []
        
        np.random.seed(42)
        for user, items in user_items.items():
            if len(items) == 1:
                train_users.append(user)
                train_items.extend(items)
            else:
                n_val = max(1, int(len(items) * val_ratio))
                val_indices = np.random.choice(len(items), n_val, replace=False)
                
                for i, item in enumerate(items):
                    if i in val_indices:
                        val_users.append(user)
                        val_items.append(item)
                    else:
                        train_users.append(user)
                        train_items.append(item)
        
        self.trainUser = np.array(train_users)
        self.trainItem = np.array(train_items)
        self.valUser = np.array(val_users)
        self.valItem = np.array(val_items)
        self._valDataSize = len(val_users)

    def _build_dict(self, users, items):
        """Build user->items dictionary"""
        data_dict = {}
        for user, item in zip(users, items):
            if user not in data_dict:
                data_dict[user] = []
            data_dict[user].append(item)
        return data_dict

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item


class Loader(BasicDataset):
    """General dataset loader"""

    def __init__(self, config=world.config, path="../data/gowalla"):
        print(f'Loading {path}...')
        self.path = path
        val_ratio = config.get('val_ratio', 0.1)
        
        original_train_data = self._load_train_data(path + '/train.txt')
        test_data = self._load_test_data(path + '/test.txt')
        self.testUser, self.testItem = test_data
        
        self._create_train_val_split(original_train_data, val_ratio)
        
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item)
        )
        
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self._testDict = self._build_dict(self.testUser, self.testItem)
        self._valDict = self._build_dict(self.valUser, self.valItem) if hasattr(self, 'valUser') else {}

    def _load_train_data(self, train_file):
        """Load training data"""
        train_data = []
        self.n_user = 0
        self.m_item = 0
        
        with open(train_file) as f:
            for line in f:
                if len(line.strip()) > 0:
                    parts = line.strip().split(' ')
                    user = int(parts[0])
                    items = [int(i) for i in parts[1:]]
                    
                    for item in items:
                        train_data.append((user, item))
                        self.n_user = max(self.n_user, user)
                        self.m_item = max(self.m_item, item)
        
        self.n_user += 1
        self.m_item += 1
        return train_data

    def _load_test_data(self, test_file):
        """Load test data"""
        test_users, test_items = [], []
        
        with open(test_file) as f:
            for line in f:
                if len(line.strip()) > 0:
                    parts = line.strip().split(' ')
                    user = int(parts[0])
                    items = [int(i) for i in parts[1:]]
                    
                    test_users.extend([user] * len(items))
                    test_items.extend(items)
                    
                    self.n_user = max(self.n_user, user)
                    for item in items:
                        self.m_item = max(self.m_item, item)
        
        return np.array(test_users), np.array(test_items)

    def _create_train_val_split(self, original_train_data, val_ratio):
        """Split training data"""
        user_items = {}
        for user, item in original_train_data:
            if user not in user_items:
                user_items[user] = []
            user_items[user].append(item)
        
        train_users, train_items = [], []
        val_users, val_items = [], []
        
        np.random.seed(42)
        for user, items in user_items.items():
            if len(items) == 1:
                train_users.append(user)
                train_items.extend(items)
            else:
                n_val = max(1, int(len(items) * val_ratio))
                val_indices = np.random.choice(len(items), n_val, replace=False)
                
                for i, item in enumerate(items):
                    if i in val_indices:
                        val_users.append(user)
                        val_items.append(item)
                    else:
                        train_users.append(user)
                        train_items.append(item)
        
        self.trainUser = np.array(train_users)
        self.trainItem = np.array(train_items)
        self.valUser = np.array(val_users)
        self.valItem = np.array(val_items)
        self._valDataSize = len(val_users)

    def _build_dict(self, users, items):
        """Build user->items dictionary"""
        data_dict = {}
        for user, item in zip(users, items):
            if user not in data_dict:
                data_dict[user] = []
            data_dict[user].append(item)
        return data_dict

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item