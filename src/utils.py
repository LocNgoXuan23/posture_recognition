import yaml
import json

def get_config(yaml_file='./config.yml'):
    with open(yaml_file, 'r') as file:
        cfgs = yaml.load(file, Loader=yaml.FullLoader)
    return cfgs

def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def get_label(path):
    lookup = {str(k): k for k in range(1,24)}
    try:
        return lookup[path.split('/')[-2]] - 1
    except:
        return None