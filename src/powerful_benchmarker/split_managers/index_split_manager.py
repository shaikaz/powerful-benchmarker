from ..utils import dataset_utils as d_u, common_functions as c_f
from .base_split_manager import BaseSplitManager
import itertools
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import logging

class IndexSplitManager(BaseSplitManager):
    def __init__(self,
        num_training_partitions,
        num_training_sets,
        test_size=None,
        test_start_idx=None,
        shuffle = False,
        random_seed = None,
        helper_split_manager = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_training_partitions = num_training_partitions
        self.num_training_sets = num_training_sets
        self.test_size = test_size
        self.test_start_idx = test_start_idx
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.helper_split_manager = helper_split_manager

    def split_assertions(self):
        super().split_assertions()
        for t_type in self.split_scheme_holder.get_transform_types():
            self.assert_across("split_scheme_names", "disjoint", transform_types=[t_type], split_names=["val"])

    def get_trainval_and_test(self, dataset, input_list):
        trainval_set, test_set, _, _ = train_test_split(input_list, 
                                                        input_list, 
                                                        test_size=self.test_size, 
                                                        shuffle=self.shuffle, 
                                                        random_state=self.random_seed)
        return trainval_set, test_set

    def get_kfold_generator(self, dataset, trainval_set):
        kfolder = KFold(n_splits=self.num_training_partitions, shuffle=self.shuffle, random_state=self.random_seed)
        return kfolder.split(trainval_set)

    def get_list_for_splitting(self, dataset):
        return np.arange(len(dataset))

    def convert_to_subset_idx(self, dataset, split_from_kfolder):
        return split_from_kfolder

    def get_trainval_and_test_from_helper_split_manager(self, dataset):
        test_set = self.helper_split_manager.get_dataset("eval", "test").indices
        trainval_set = np.array(list(set(range(len(dataset))) - set(test_set)))
        return trainval_set, test_set

    def _create_split_schemes(self, datasets):
        # the assumption is all datasets are the same
        sample_dataset = c_f.first_val_of_dict(c_f.first_val_of_dict(datasets))

        if self.helper_split_manager is not None:
            logging.info("***Using helper_split_manager to create the test set***")
            self.helper_split_manager.create_split_schemes(datasets)
            trainval_set, test_set = self.get_trainval_and_test_from_helper_split_manager(sample_dataset)
        else:
            list_for_splitting = np.array(self.get_list_for_splitting(sample_dataset))
            if not self.shuffle:
                num_idx = len(list_for_splitting)
                test_start = int(self.test_start_idx*num_idx)
                test_size = int(self.test_size*num_idx)
                list_for_splitting = np.roll(list_for_splitting, test_start - test_size)
            trainval_set, test_set = self.get_trainval_and_test(sample_dataset, list_for_splitting)

        trainval_idx_tuples = self.get_kfold_generator(sample_dataset, trainval_set)
        return d_u.create_subset_datasets_from_indices(datasets, 
                                                        trainval_idx_tuples,
                                                        trainval_set, 
                                                        test_set, 
                                                        self.get_split_scheme_name,
                                                        self.num_training_sets,
                                                        self.convert_to_subset_idx)


    def get_test_set_name(self):
        if self.helper_split_manager is not None:
            return self.helper_split_manager.get_test_set_name()
        else:
            shuffle_string = 'Shuffled' if self.shuffle else ''
            test_size = int(self.test_size*100)
            test_start_idx = int(self.test_start_idx*100)
            return '{}Test{:02d}_{:02d}'.format(shuffle_string, test_size, test_start_idx)

    def get_base_split_scheme_name(self):
        shuffle_string = 'Shuffled' if self.shuffle else ''
        return '{}_{}Partitions{}'.format(self.get_test_set_name(), shuffle_string, self.num_training_partitions)

    def get_split_scheme_name(self, partition):
        return "{}_{}".format(self.get_base_split_scheme_name(), partition)
