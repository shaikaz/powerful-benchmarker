import sys
import record_keeper as record_keeper_package
from easy_module_attribute_getter import PytorchGetter
import datasets
import copy
from utils import common_functions as c_f, split_manager
from pytorch_metric_learning import trainers, losses, miners, samplers, testers
from torch.utils.tensorboard import SummaryWriter
import torch.nn
import torch
import architectures
import os
import shutil


class BaseAPIParser:

    def __init__(self, args):
        os.environ["TORCH_HOME"] = args.pytorch_home
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.experiment_folder = args.experiment_folder
        self.pytorch_getter = PytorchGetter(use_pretrainedmodels_package=True)
        self.pytorch_getter.register('model', architectures.misc_models)
        self.pytorch_getter.register('loss', losses)
        self.pytorch_getter.register('miner', miners)
        self.pytorch_getter.register('sampler', samplers)
        self.pytorch_getter.register('trainer', trainers)
        self.pytorch_getter.register('tester', testers)
        self.pytorch_getter.register('dataset', datasets)

        _model_folder = "%s/%s/saved_models"
        _pkl_folder = "%s/%s/saved_pkls"
        _tensorboard_folder = "%s/%s/tensorboard_logs"
        self.sub_experiment_dirs = [
            _model_folder,
            _pkl_folder,
            _tensorboard_folder
        ]

    def run(self):
        if self.beginning_of_training():
            self.make_dir()
        self.save_config_files()
        self.set_split_manager()
        self.make_sub_experiment_dirs()
        for split_scheme_name in self.split_manager.split_scheme_names:
            self.split_manager.set_curr_split_scheme(split_scheme_name)
            self.set_curr_folders()
            self.set_models_optimizers_losses()
            self.eval() if self.args.run_eval_only else self.train()

    def beginning_of_training(self):
        return (not self.args.resume_training) and (not self.args.run_eval_only)

    def make_dir(self):
        if os.path.isdir(self.experiment_folder):
            print("Experiment name already taken!")
            sys.exit()
        c_f.makedir_if_not_there(self.experiment_folder)

    def make_sub_experiment_dirs(self):
        for s in self.sub_experiment_dirs:
            for r in self.split_manager.split_scheme_names:
                c_f.makedir_if_not_there(s % (self.experiment_folder, r))

    def set_curr_folders(self):
        r = self.split_manager.curr_split_scheme_name
        self.model_folder, self.pkl_folder, self.tensorboard_folder = [
            s % (self.experiment_folder, r) for s in self.sub_experiment_dirs
        ]

    def save_config_files(self):
        c_f.save_config_files(self.args.place_to_save_configs, self.args.dict_of_yamls, self.args.resume_training)
        delattr(self.args, "dict_of_yamls")
        delattr(self.args, "place_to_save_configs")

    def set_optimizers(self):
        self.optimizers, self.lr_schedulers, self.gradient_clippers = {}, {}, {}
        for k, v in self.args.optimizers.items():
            basename = k.replace("_optimizer", '')
            for possible_params in [self.models, self.loss_funcs]:
                if basename in possible_params:
                    param_source = possible_params[basename]
                    break
            o, s, g = self.pytorch_getter.get_optimizer(param_source, yaml_dict=v)
            print("%s\n%s" % (k, o))
            if o is not None: self.optimizers[k] = o
            if s is not None: self.lr_schedulers[k + "_scheduler"] = s
            if g is not None: self.gradient_clippers[k + "_grad_clipper"] = g

    def set_split_manager(self):
        chosen_dataset, dataset_params = self.pytorch_getter.get("dataset", yaml_dict=self.args.dataset, return_uninitialized=True)
        dataset_params["dataset_root"] = self.args.dataset_root

        if self.args.split_schemes[0] == "fixed":
            input_dataset_splits = {
                "fixed": {
                    k: (chosen_dataset(split=k, **dataset_params), None)
                    for k in self.args.split_schemes[1:]
                }
            }
            self.split_manager = split_manager.SplitManager(input_dataset_splits=input_dataset_splits)
        else:
            chosen_dataset = chosen_dataset(**dataset_params)
            self.split_manager = split_manager.SplitManager(dataset=chosen_dataset, 
                                                            split_scheme_names=self.args.split_schemes, 
                                                            num_variants_per_split_scheme=self.args.num_variants_per_split_scheme)

    def set_transforms(self):
        try:
            model_transform_properties = {k:getattr(self.models["trunk"], k) for k in ["mean", "std", "input_space", "input_range"]}
        except:
            model_transform_properties = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        transforms = {}
        for k, v in self.args.transforms.items():
            transforms[k] = self.pytorch_getter.get_composed_img_transform(v, **model_transform_properties)
        self.split_manager.set_transforms(transforms["train"], transforms["eval"])

    def set_dataset(self):
        self.split_manager.set_curr_split("train", is_training=True)
        self.dataset = self.split_manager.dataset

    def get_embedder_model(self, model_type, input_size=None, output_size=None):
        model, model_args = self.pytorch_getter.get("model", yaml_dict=model_type, return_uninitialized=True)
        model_args = copy.deepcopy(model_args)
        if model == architectures.misc_models.MLP:
            if input_size:
                model_args["layer_sizes"].insert(0, input_size)
            if output_size:
                model_args["layer_sizes"].append(output_size)
        model = model(**model_args)
        print("EMBEDDER MODEL ", model)
        return model

    def get_trunk_model(self, model_type, force_pretrained=False):
        additional_params = {}
        if force_pretrained:
            additional_params["pretrained"] = 'imagenet'
        model = self.pytorch_getter.get("model", yaml_dict=model_type, additional_params=additional_params)
        self.base_model_output_size = c_f.get_last_linear(model).in_features
        c_f.set_last_linear(model, architectures.misc_models.Identity())
        return model

    def get_mining_function(self, miner_type):
        if miner_type:
            miner, miner_params = self.pytorch_getter.get("miner", yaml_dict=miner_type, return_uninitialized=True)
            miner_params = copy.deepcopy(miner_params)
            if "loss_function" in miner_params:
                loss_args = miner_params["loss_function"]
                miner_params["loss_function"] = self.get_loss_function(loss_args)
            if "mining_function" in miner_params:
                miner_args = miner_params["mining_function"]
                miner_params["mining_function"] = self.get_mining_function(miner_args)
            miner = miner(**miner_params)
            return miner
        return None

    def set_sampler(self):
        self.sampler = self.pytorch_getter.get("sampler", yaml_dict=self.args.sampler, additional_params={"labels_to_indices":self.split_manager.labels_to_indices})

    def set_loss_function(self):
        self.loss_funcs = self.pytorch_getter.get_multiple("loss", yaml_dict=self.args.loss_funcs)

    def set_mining_function(self):
        self.mining_funcs = {}
        for k, v in self.args.mining_funcs.items():
            self.mining_funcs[k] = self.get_mining_function(v)

    def model_getter_dict(self):
        return {"trunk": lambda model_type: self.get_trunk_model(model_type),
                "embedder": lambda model_type: self.get_embedder_model(model_type, input_size=self.base_model_output_size)}

    def set_model(self):
        self.models = {}
        for k, v in self.model_getter_dict().items():
            self.models[k] = v(self.args.models[k])

    def load_model_for_eval(self, resume_epoch=None):
        pretrained = resume_epoch == -1
        trunk_model = self.get_trunk_model(self.args.models["trunk"], force_pretrained=pretrained)
        embedder_model = self.get_embedder_model(self.args.models["embedder"], self.base_model_output_size)
        if not pretrained:
            c_f.load_dict_of_models(
                {"trunk": trunk_model, "embedder": embedder_model},
                resume_epoch,
                self.model_folder,
            )
        else:
            embedder_model = architectures.misc_models.Identity()
        return trunk_model, embedder_model

    def eval_model(self, epoch, load_model=False):
        dataset_dict = self.split_manager.get_dataset_dict(is_training=False)
        if load_model:
            trunk_model, embedder_model = self.load_model_for_eval(resume_epoch=epoch)
        else:
            trunk_model, embedder_model = self.models["trunk"], self.models["embedder"]
        trunk_model.to(self.device)
        embedder_model.to(self.device)
        self.tester_obj.test(dataset_dict, epoch, trunk_model, embedder_model)

    def save_stuff_and_maybe_eval(self, epoch):
        if epoch % self.args.save_interval == 0:
            for obj_dict in [self.models, self.optimizers, self.lr_schedulers, self.loss_funcs]:
                c_f.save_dict_of_models(obj_dict, epoch, self.model_folder)
            if not self.args.skip_eval:
                if epoch == self.args.save_interval and self.args.check_pretrained_accuracy:
                    self.eval_model(-1, load_model=True)
                self.eval_model(epoch)
            self.pickler_and_csver.save_records()

    def set_record_keeper(self):
        self.pickler_and_csver = record_keeper_package.PicklerAndCSVer(self.pkl_folder)
        self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_folder)
        self.record_keeper = record_keeper_package.RecordKeeper(
            self.tensorboard_writer, self.pickler_and_csver, ["record_these", "learnable_param_names"])

    def maybe_load_models_and_records(self):
        resume_epoch = 0
        self.pickler_and_csver.load_records()
        if self.args.resume_training:
            resume_epoch = c_f.latest_version(self.model_folder, "/trunk_*.pth")
            for obj_dict in [self.models, self.optimizers, self.lr_schedulers, self.loss_funcs]:
                c_f.load_dict_of_models(obj_dict, resume_epoch, self.model_folder)
        return resume_epoch

    def set_models_optimizers_losses(self):
        self.set_dataset()
        self.set_model()
        self.set_transforms()
        self.set_sampler()
        self.set_dataparallel()
        self.set_loss_function()
        self.set_mining_function()
        self.set_optimizers()
        self.set_record_keeper()
        self.tester_obj = self.pytorch_getter.get("tester", self.args.testing_method, self.get_tester_kwargs())
        self.epoch = self.maybe_load_models_and_records() + 1

    def get_tester_kwargs(self):
        return {
            "reference_set": self.args.eval_reference_set,
            "normalize_embeddings": self.args.eval_normalize_embeddings,
            "use_trunk_output": self.args.eval_use_trunk_output,
            "batch_size": self.args.eval_batch_size,
            "split_for_best_epoch": "val" if "val" in self.split_manager.curr_split_scheme.keys() else "test",
            "metric_for_best_epoch": self.args.eval_metric_for_best_epoch,
            "data_device": self.device,
            "dataloader_num_workers": self.args.eval_dataloader_num_workers,
            "record_keeper": self.record_keeper
        }

    def get_trainer_kwargs(self):
        return {
            "models": self.models,
            "optimizers": self.optimizers,
            "batch_size": self.args.batch_size,
            "sampler": self.sampler,
            "collate_fn": None,
            "loss_funcs": self.loss_funcs,
            "mining_funcs": self.mining_funcs,
            "dataset": self.dataset,
            "data_device": self.device,
            "record_keeper": self.record_keeper,
            "num_epochs": self.args.num_epochs_train,
            "iterations_per_epoch": self.args.iterations_per_epoch,
            "label_mapper": self.split_manager.map_labels,
            "lr_schedulers": self.lr_schedulers,
            "gradient_clippers": self.gradient_clippers,
            "freeze_trunk_batchnorm": self.args.freeze_batchnorm,
            "label_hierarchy_level": self.args.label_hierarchy_level,
            "dataloader_num_workers": self.args.dataloader_num_workers,
            "loss_weights": getattr(self.args, "loss_weights", None) 
        }

    def set_dataparallel(self):
        for k, v in self.models.items():
            self.models[k] = torch.nn.DataParallel(v)

    def set_devices(self):
        for obj_dict in [self.models, self.loss_funcs, self.mining_funcs]:
            for v in obj_dict.values():
                v.to(self.device)
        for v in self.optimizers.values():
            c_f.move_optimizer_to_gpu(v, self.device)

    def train(self):
        self.trainer = self.pytorch_getter.get("trainer", self.args.training_method, self.get_trainer_kwargs())
        self.trainer.num_epochs = self.epoch - 1
        while self.epoch <= self.args.num_epochs_train:
            self.set_devices()
            self.trainer.num_epochs += self.args.save_interval
            self.trainer.epoch = self.epoch
            self.trainer.train()
            self.epoch = self.trainer.epoch - 1
            self.save_stuff_and_maybe_eval(self.epoch)
            self.epoch += 1

    def eval(self):
        if self.args.run_eval_only == ["all"]:
            epochs_list = range(self.args.save_interval, self.args.num_epochs_train + 1, self.args.save_interval)
        else:
            epochs_list = [int(x) for x in self.args.run_eval_only]
        for epoch in epochs_list:
            self.eval_model(epoch, load_model=True)
            self.pickler_and_csver.save_records()
