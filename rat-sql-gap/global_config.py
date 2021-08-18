import os
import json


class Config(object):
    def __init__(self):
        pass

    def set(self, k, v):
        setattr(self, k, v)


global_config_dict = {
    'reg_loss_weight': 1,
    'tc_loss_weight': 8,
    'contrast_loss_weight': 10
}

if os.path.exists('global_config.json'):
    params = json.load(open('global_config.json', 'r', encoding='utf-8'))
    global_config_dict.update(params)
else:
    print("\033[1;31m[Warning] Configuration file `global_config.json` not found\033[0m")

global_config = Config()
for k, v in global_config_dict.items():
    global_config.set(k, v)


if __name__ == '__main__':
    json.dump(global_config_dict, open('global_config.json', 'w', encoding='utf-8'))
