# DependencyParsing
A transition based dependency parser.

A TensorFlow (>= 0.12.1) implementation of a transition-based dependency parser with a similar architecture in the paper "A Fast and Accurate Dependency Parser using Neural Networks" in EMNLP 2014. Bidirectional LSTM based character-level embedding is also supported and the character-level embedding would be concatenated with word embedding to compose the final embedding vector for each word.

Training and evaluation file should be in CoNLL format. 

# Help
python DependencyParsing.py -h

# Training
python DependencyParsing.py -T train -d "data_directory_path" -t training_filename -v dev_filename -p "pretrained_word_embedding_path" -e 128 -H 100 -k 0.5 -l 0.01 -b 20 -i 3000 --decay 0.98

python DependencyParsing.py -T train -d "data_directory_path" -t training_filename -v dev_filename -p "pretrained_word_embedding_path" -e 128 -H 100 --use_chars -E 50 --hidden_size_char 50 -k 0.5 -l 0.01 -b 20 -i 3000 --decay 0.98

# Evaluation
python DependencyParsing.py -T eval --model_dir "model_path" --eval_filepath "eval_filepath"

# Interactive prediction
python DependencyParsing.py -T online --model_dir "model_path"
