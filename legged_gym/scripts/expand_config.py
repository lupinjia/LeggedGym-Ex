import inspect
from legged_gym.envs.go2.go2_dreamwaq_lora_env.go2_dreamwaq_lora_env_config import Go2DreamwaqLoraCfg, Go2DreamwaqLoraCfgPPO
from legged_gym.utils import *
import argparse

def is_config_class(obj):
    return inspect.isclass(obj)

def get_class_dict(cls):
    result = {}
    for base in reversed(cls.__mro__):
        if base is object:
            continue
        for k, v in base.__dict__.items():
            if k.startswith("__"):
                continue
            result[k] = v
    return result


def flatten_config_class(cls):
    output = {}
    class_dict = get_class_dict(cls)

    for key, value in class_dict.items():
        if inspect.isclass(value):
            output[key] = flatten_config_class(value)
        elif not callable(value):
            output[key] = value
    return output

def dict_to_class(name, d, indent=0):
    lines = []
    ind = " " * indent

    lines.append(f"{ind}class {name}:")

    if not d:
        lines.append(f"{ind}    pass")
        return "\n".join(lines)

    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(dict_to_class(k, v, indent + 4))
        else:
            lines.append(f"{ind}    {k} = {repr(v)}")

    return "\n".join(lines)


def string_expanded_class(cls, class_name=None):
    flat = flatten_config_class(cls)
    class_name = (class_name or cls.__name__) + "(BaseConfig)"
    code = dict_to_class(class_name, flat)
    return code

def print_expanded_class(cls, class_name=None):
    flat = flatten_config_class(cls)
    class_name = (class_name or cls.__name__) + "(BaseConfig)"
    code = dict_to_class(class_name, flat)
    print(code)

def reconstruct_config_file(env_cfg_class, train_cfg_class):
    s = "from legged_gym.envs.base.base_config import BaseConfig\n"
    s += "\n"
    s += string_expanded_class(env_cfg_class)
    s += "\n"
    s += "\n"
    s += string_expanded_class(train_cfg_class)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='go2', help="task name(s)")
    args = parser.parse_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    print(reconstruct_config_file(env_cfg.__class__, train_cfg.__class__))