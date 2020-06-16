#! /usr/bin/env python3

import collections
import operator
import errno
import glob
import os
import itertools

import numpy as np
import torch
import yaml
from easy_module_attribute_getter import utils as emag_utils
import logging
import inspect
import pytorch_metric_learning.utils.common_functions as pml_cf
import datetime
import sqlite3
import tqdm
import tarfile, zipfile

CONFIG_DIFF_BASE_FOLDER_NAME = "resume_training_config_diffs_"


def move_optimizer_to_gpu(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def makedir_if_not_there(dir_name):
    try:
        os.makedirs(dir_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_yaml(fname):
    with open(fname, 'r') as f:
        loaded_yaml = yaml.safe_load(f)
    return loaded_yaml


def write_yaml(fname, input_dict, open_as):
    with open(fname, open_as) as outfile:
        yaml.dump(input_dict, outfile, default_flow_style=False, sort_keys=False)


def latest_sub_experiment_epochs(sub_experiment_dir_dict):
    latest_epochs = {}
    for sub_experiment_name, folders in sub_experiment_dir_dict.items():
        model_folder = folders["models"]
        latest_epochs[sub_experiment_name], _ = pml_cf.latest_version(model_folder)
    return latest_epochs


def get_sorted_config_diff_folders(config_folder):
    full_base_path = os.path.join(config_folder, CONFIG_DIFF_BASE_FOLDER_NAME)
    config_diff_folder_names = glob.glob("%s*"%full_base_path) 
    latest_epochs = []
    if len(config_diff_folder_names) > 0:
        for c in config_diff_folder_names:
            latest_epochs.append([c]+[int(x) for x in c.replace(full_base_path,"").split('_')])
        num_training_sets = len(latest_epochs[0])-1
        latest_epochs = sorted(latest_epochs, key=operator.itemgetter(*list(range(1, num_training_sets+1))))
        return [x[0] for x in latest_epochs], [x[1:] for x in latest_epochs]
    return [], []

def get_all_resume_training_config_diffs(config_folder, split_manager):
    config_diffs, latest_epochs = get_sorted_config_diff_folders(config_folder)
    if len(config_diffs) == 0:
        return {}
    split_scheme_names = [split_manager.get_split_scheme_name(i) for i in range(len(latest_epochs[0]))]
    resume_training_dict = {}
    for i, k in enumerate(config_diffs):
        resume_training_dict[k] = {split_scheme:epoch for (split_scheme,epoch) in zip(split_scheme_names, latest_epochs[i])}
    return resume_training_dict


def save_config_files(config_folder, dict_of_yamls, resume_training, latest_epochs):
    makedir_if_not_there(config_folder)
    new_dir = None
    existing_config_diff_folders, _ = get_sorted_config_diff_folders(config_folder)

    for config_name, config_dict in dict_of_yamls.items():
        fname = os.path.join(config_folder, '%s.yaml'%config_name)
        if not resume_training:
            write_yaml(fname, config_dict, 'w')
        else:
            curr_yaml = load_yaml(fname)
            for config_diff_folder in existing_config_diff_folders:
                config_diff = os.path.join(config_diff_folder, '%s.yaml'%config_name)
                if os.path.isfile(config_diff):
                    curr_yaml = emag_utils.merge_two_dicts(curr_yaml, load_yaml(config_diff), max_merge_depth=float('inf'))

            yaml_diff = {}
            for k, v in config_dict.items():
                if (k not in curr_yaml) or (v != curr_yaml[k]):
                    yaml_diff[k] = v

            if yaml_diff != {}:
                new_dir = os.path.join(config_folder, CONFIG_DIFF_BASE_FOLDER_NAME + '_'.join([str(epoch) for epoch in latest_epochs]))
                makedir_if_not_there(new_dir)
                fname = os.path.join(new_dir, '%s.yaml' %config_name)
                write_yaml(fname, yaml_diff, 'a')


def get_last_linear(input_model, return_name=False):
    for name in ["fc", "last_linear"]:
        last_layer = getattr(input_model, name, None)
        if last_layer:
            if return_name:
                return last_layer, name
            return last_layer

def set_last_linear(input_model, set_to):
    setattr(input_model, get_last_linear(input_model, return_name=True)[1], set_to)


def check_init_arguments(input_obj, str_to_check):
    obj_stack = [input_obj]
    while len(obj_stack) > 0:
        curr_obj = obj_stack.pop()
        obj_stack += list(curr_obj.__bases__)
        if str_to_check in str(inspect.signature(curr_obj.__init__)):
            return True
    return False


def try_getting_db_count(record_keeper, table_name):
    try:
        len_of_existing_record = record_keeper.query("SELECT count(*) FROM %s"%table_name, use_global_db=False)[0]["count(*)"] 
    except sqlite3.OperationalError:
        len_of_existing_record = 0
    return len_of_existing_record


def get_datetime():
    return datetime.datetime.now()


def extract_progress(compressed_obj):
    logging.info("Extracting dataset")
    if isinstance(compressed_obj, tarfile.TarFile):
        iterable = compressed_obj
        length = len(compressed_obj.getmembers())
    elif isinstance(compressed_obj, zipfile.ZipFile):
        iterable = compressed_obj.namelist()
        length = len(iterable)
    for member in tqdm.tqdm(iterable, total=length):
        yield member


def if_str_convert_to_singleton_list(input):
    if isinstance(input, str):
        return [input]
    return input


def first_val_of_dict(input):
    return input[list(input.keys())[0]]


def get_attr_and_try_as_function(input_object, input_attr):
    attr = getattr(input_object, input_attr)
    try:
        return attr()
    except TypeError:
        return attr