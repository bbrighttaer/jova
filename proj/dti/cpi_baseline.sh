python cpi_baseline.py --dataset_name kiba --dataset_file ../../data/KIBA_data/restructured_unique.csv --prot_desc_path ../../data/KIBA_data/prot_desc.csv --model_dir ./model_dir/kiba --filter_threshold 6 --split warm --split cold_drug --split cold_target --view gnn --fold_num 5 --prot_profile ../../data/protein/proteins.profile --prot_vocab ../../data/protein/protein_words_dict.pkl