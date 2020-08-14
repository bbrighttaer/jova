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
                if '}.json' in f:
                    long_filename_dict = json.loads(f.split(".j")[0])
                    new_filename = '_'.join([long_filename_dict['model_family'],
                                             long_filename_dict['dataset'],
                                             long_filename_dict['split'],
                                             long_filename_dict['date'],
                                             '.json'])
                    try:
                        with open(os.path.join(root, f), 'r') as data_file:
                            json_data = json.load(data_file)
                        json_data['metadata'] = long_filename_dict
                        with open(os.path.join(root, new_filename), 'w') as data_file:
                            json.dump(dict(json_data), data_file)
                        os.remove(os.path.join(root, f))
                        success_count += 1
                    except Exception as e:
                        print(os.path.join(root, f))
                        print(str(e))
                elif '}.csv' in f:
                    long_filename_dict = json.loads(f.split(".c")[0])
                    new_filename = '_'.join([long_filename_dict['model_family'],
                                             long_filename_dict['dataset'],
                                             long_filename_dict['model_ds'],
                                             long_filename_dict['cviews'],
                                             long_filename_dict['pviews'],
                                             long_filename_dict['split'],
                                             long_filename_dict['date'],
                                             '.csv'])
                    try:
                        df = pd.read_csv(os.path.join(root, f))
                        df.to_csv(os.path.join(root, new_filename), index=False)
                        os.remove(os.path.join(root, f))
                        success_count += 1
                    except Exception as e:
                        print(os.path.join(root, f))
    print(success_count)
