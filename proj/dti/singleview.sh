python singleview.py --dataset_name metz --dataset_file ../../data/metz_data/restructured_unique.csv --prot_desc_path ../../data/metz_data/prot_desc.csv --model_dir ./model_dir/metz --filter_threshold 1 --split cold_target --split cold_drug --comp_view weave --prot_view rnn --fold_num 5 --prot_profile ../../data/protein/proteins.profile --prot_vocab ../../data/protein/protein_words_dict.pkl -mp