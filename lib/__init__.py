import argparse
import json

def get_conf(name):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=f'./configs/{name}.json')
    options = parser.parse_args()
    config = json.load(open(options.config))

    return config