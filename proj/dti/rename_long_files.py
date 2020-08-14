# Author: bbrighttaer
# Project: jova
# Date: 8/14/20
# Time: 11:08 PM
# File: rename_long_files.py


from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import json

if __name__ == '__main__':
    json_dir = './analysis'
    json_files = []
    for root, directories, file in os.walk(json_dir):
        if len(file) > 0:
            for f in file:
                if '}.json' in f:
                    long_filename_dict = json.loads(f.split(".j")[0])
                    new_filename = '_'.join([long_filename_dict['model_family'],
                                             long_filename_dict['dataset'],
                                             long_filename_dict['split'],
                                             long_filename_dict['date'],
                                             '.json'])
                    with open(os.path.join(root, f), 'r') as data_file:
                        print(root, data_file)
                        json_data = json.load(data_file)
                    json_data['metadata'] = long_filename_dict
                    with open(os.path.join(root, new_filename), 'w') as data_file:
                        json.dump(dict(json_data), data_file)
                    os.remove(os.path.join(root, f))

