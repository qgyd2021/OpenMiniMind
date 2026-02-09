#!/usr/bin/python3
# -*- coding: utf-8 -*-
import copy
import os
from typing import Any, Dict, Union

import yaml


CONFIG_FILE = "config.yaml"


class PretrainedConfig(object):
    def __init__(self, **kwargs):
        pass

    @classmethod
    def _dict_from_yaml_file(cls, yaml_file: Union[str, os.PathLike]):
        with open(yaml_file, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return config_dict

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike]
    ) -> Dict[str, Any]:
        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_FILE)
        else:
            config_file = pretrained_model_name_or_path
        config_dict = cls._dict_from_yaml_file(config_file)
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):
        for k, v in kwargs.items():
            if k in config_dict.keys():
                config_dict[k] = v
        config = cls(**config_dict)
        return config

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ):
        config_dict = cls.get_config_dict(pretrained_model_name_or_path)
        return cls.from_dict(config_dict, **kwargs)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_yaml_file(self, yaml_file_path: Union[str, os.PathLike]):
        config_dict = self.to_dict()

        with open(yaml_file_path, "w", encoding="utf-8") as writer:
            yaml.safe_dump(config_dict, writer)


if __name__ == '__main__':
    pass
