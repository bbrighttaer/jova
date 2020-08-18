# Author: bbrighttaer
# Project: jova
# Date: 8/14/20
# Time: 11:08 PM
# File: rename_long_files.py


from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
import pandas as pd

from tqdm import tqdm

if __name__ == '__main__':
    json_dir = './analysis'
    json_files = []
    success_count = 0
    for root, directories, file in tqdm(os.walk(json_dir)):
        if len(file) > 0:
            for f in file:
                if '_.json' in f:
                    try:
                        with open(os.path.join(root, f), 'r') as data_file:
                            json_data = json.load(data_file)
                        long_filename_dict = json_data['metadata']
                        new_filename = '_'.join([long_filename_dict['model_family'],
                                                 long_filename_dict['dataset'],
                                                 long_filename_dict['split'],
                                                 long_filename_dict['mode'],
                                                 long_filename_dict['date']])
                        new_json_data = {new_filename: json_data[list(json_data.keys())[0]],
                                         'metadata': long_filename_dict}
                        with open(os.path.join(root, new_filename + '.json'), 'w') as data_file:
                            json.dump(dict(new_json_data), data_file)
                        # os.remove(os.path.join(root, f))
                        success_count += 1
                    except Exception as e:
                        print(os.path.join(root, f))
                        print(str(e))
                    os.remove(os.path.join(root, f))
    print(success_count)
