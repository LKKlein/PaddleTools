import os
import json
import yaml
import argparse

from paddletools import logger


def str2bool(value):
    """ String to Boolean """
    # because argparse does not support to parse "True, False" as python
    # boolean directly
    return value.lower() in ("true", "t", "1")


def str2list(value):
    """
    String to list, with comma split.
    Now, only support axis-1 list.
    """
    return value.split(",")


class ArgumentGroup(object):
    """ Argument Class """

    def __init__(self, parser, title, desc):
        self._group = parser.add_argument_group(title=title, description=desc)

    def add_arg(self, name, dtype, default, help, **kwargs):
        """ Add argument """
        dtype = str2bool if dtype == bool else dtype
        dtype = str2list if dtype == list else dtype
        self._group.add_argument(
            "--" + name,
            default=default,
            type=dtype,
            help=help + " Default: %(default)s.",
            **kwargs)


class PDConfig(object):
    """ A high-level api for handling argument configs. """

    def __init__(self):
        """ Init function for PDConfig. """
        self.args = None
        self.arg_config = {}
        self.custom_config = {}

        self.parser = argparse.ArgumentParser()
        self._init_preset_args()
        self.arg_groups = {}

    def _init_preset_args(self):
        model_args_names = {
            "init_checkpoint": [str, None, "init checkpoint to resume training from."],
            "checkpoints": [str, "checkpoints", "Path to save checkpoints"],
            "pretrain": [str, None, "Path to pretrain weights"]
        }

        train_args_names = {
            "epoch": [int, 100, "Number of epoches for training."],
            "save_steps": [int, 10000, "The steps interval to save checkpoints."],
            "valid_steps": [int, 1000, "The steps interval to eval model performance."],
            "lr": [float, 0.002, "The Learning rate value for training."],
            "momentum": [float, 0.9, "The momentum for Momentum Optimizer."],
            "freeze_backbone": [bool, False, "Whether to freeze backbone."],
            "freeze_norm": [float, False, "Whether to freeze bn norm params."]
        }

        log_args_names = {
            "skip_steps": [int, 10, "The steps interval to print loss."],
            "verbose": [bool, False, "Whether to output verbose log"]
        }

        data_args_names = {
            "data_dir": [str, None, "Path to training data."],
            "batch_size": [int, 256, "Total examples number in batch for training."],
            "random_seed": [int, 0, "Random seed."],
            "num_labels": [int, 1000, "label number"],
            "train_set": [str, None, "Path to training data."],
            "eval_set": [str, None, "Path to validation data."],
            "test_set": [str, None, "Path to test data."]
        }

        run_args_names = {
            "use_cuda": [bool, True, "If set, use GPU for training."],
            "task_name": [str, None, "The name of running task."],
            "do_train": [bool, True, "Whether to perform training."],
            "do_eval": [bool, True, "Whether to perform evaluation."],
            "do_infer": [bool, True, "Whether to perform inference."],
            "do_save_inference_model": [bool, True, "Whether to save inference model"],
            "inference_model_dir": [str, None, "Path to save inference model"]
        }

        self.arg_groups_names = {
            "model": [model_args_names, "model configuration and paths."],
            "training": [train_args_names, "training options."],
            "logging": [log_args_names, "logging related"],
            "data": [data_args_names, "Data paths, and data processing options"],
            "run_type": [run_args_names, "running type options."],
            "customized": [[], "customize options"]
        }

    def load_json(self, file_path, add_to_cmd=False):
        if not os.path.exists(file_path):
            raise Warning("the json file %s does not exist." % file_path)
            return
        try:
            with open(file_path, "r") as fin:
                config = json.load(fin)
        except Exception as e:
            raise IOError("Error in parsing json config file %s" % file_path)

        if add_to_cmd:
            self._parse_config_to_cmd(config)
        else:
            self.custom_config.update(config)

    def load_yaml(self, file_path, add_to_cmd=False):
        if not os.path.exists(file_path):
            raise Warning("the yaml file %s does not exist." % file_path)
            return
        try:
            with open(file_path, "r") as fin:
                config = yaml.load(fin.read(), Loader=yaml.FullLoader)
        except Exception as e:
            raise IOError("Error in parsing yaml config file %s" % file_path)

        if add_to_cmd:
            self._parse_config_to_cmd(config)
        else:
            self.custom_config.update(config)

    def load_dict(self, config, use_cmd=True):
        if use_cmd:
            self._parse_config_to_cmd(config)
        else:
            self.custom_config.update(config)

    def _parse_config_to_cmd(self, config):
        logger.debug("cammand line only support (int, float, bool, str, list of base) type args.")
        for name, value in config.items():
            base_type = (int, float, bool, str)
            if not isinstance(value, base_type + (list,)):
                self.custom_config[name] = value
                continue
            if isinstance(value, list) and len(value) == 0:
                self.custom_config[name] = value
                continue
            if isinstance(value, list) and (not isinstance(value[0], base_type)):
                self.custom_config[name] = value
                continue

            preset_flag = False
            for group_name, group_item in self.arg_groups_names.items():
                if name in group_item[0]:
                    if group_name not in self.arg_groups:
                        self.arg_groups[group_name] = ArgumentGroup(
                            self.parser, group_name, group_item[1])
                    self.arg_groups[group_name].add_arg(
                        name, type(value), value, group_item[0][name][2])
                    preset_flag = True
                    break

            if not preset_flag:
                group_name = "customized"
                if group_name not in self.arg_groups:
                    self.arg_groups[group_name] = ArgumentGroup(
                        self.parser, group_name, self.arg_groups_names[group_name][1])
                self.arg_groups[group_name].add_arg(
                    name, type(value), value, "")

    def build(self):
        self.args = self.parser.parse_args()
        self.arg_config = vars(self.args)

    def log_arguments(self):
        logger.info("-----------  Configuration Arguments -----------")
        for arg, value in sorted(self.arg_config.items()):
            logger.info("%s: %s" % (arg, value))

        for arg, value in sorted(self.custom_config.items()):
            logger.info("%s: %s" % (arg, value))
        logger.info("------------------------------------------------")

    def add_arg(self, name, dtype, default, descrip, group="customized"):
        self.add_cmd_arg(name, dtype, default, descrip, group)

    def add_cmd_arg(self, name, dtype, default, descrip, group="customized"):
        if group not in self.arg_groups:
            self.arg_groups[group] = ArgumentGroup(
                self.parser, group, self.arg_groups_names[group][1])
        self.arg_groups[group].add_arg(name, dtype, default, descrip)

    def add_more_args(self, args):
        for name, value in args.items():
            self[name] = value

    def __setitem__(self, name, value):
        if name in self.arg_config:
            self.arg_config.pop(name)
        self.custom_config[name] = value

    def __getattr__(self, name):
        if name in self.custom_config:
            return self.custom_config[name]

        if name in self.arg_config:
            return self.arg_config[name]

        raise Warning("The argument {} is not defined.".format(name))


if __name__ == "__main__":
    config = PDConfig()
    config.add_arg("crop_size", int, 224, "image crop size to model")
    args = {
        "train_dir": "data/insects/new_train",
        "eval_dir": "data/insects/new_val",

        "anchors": [
            [19, 29], [28, 20], [25, 40],
            [31, 47], [36, 37], [41, 26],
            [47, 66], [48, 33], [67, 53]
        ],
        "anchor_masks": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        "num_classes": 7,
        "keep_topk": 100,
        "nms_thresh": 0.45,
        "score_threshold": 0.01,
        "image_shape": 608,
        "ignore_thresh": 0.7,
        "num_max_boxes": 50,

        "lr": 0.0005,
        "l2_coffe": 0.0012,
        "steps_per_epoch": 180,
        "epochs": 300,
        "momentum": 0.9,
        "warm_up_steps": 10000,

        "iters": 54000,
        "save_iter": 500,
        "log_iter": 100,
        "batch_size": 10,

        "ignore_weights": [],
        "pretrain_weights": "models/yolov3_resnet50vd_dcn_best_model",
        "save_dir": "models/",
        "save_best": "models/yolov3_resnet50vd_dcn_best_model_2",

        "freeze_backbone": False,
        "freeze_route": [],
        "freeze_block": [],
        "freeze_norm": False,
        "use_label_smooth": True,
        "with_mixup": True,
        "drop_block": True,

        "map_type": "11point",
        "shuffle_images": True,
        "use_cuda": True,
        "_eval": True
    }
    config.load_dict(args)
    config.build()
    config.print_arguments()
