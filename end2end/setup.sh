#!/bin/bash
sudo apt-get install -y graphviz libgraphviz-dev
python create_directories.py
echo "Downloading the dependencies"
pip install -r requirements.txt
echo "Dowloading and unzipping glove embeddings"
cd glove_directory
wget "http://nlp.stanford.edu/data/glove.6B.zip"
unzip glove.6B.zip
cd ..
echo "Creating,training and evaluating models"
python visualize.py
python simple_nn.py
python cnn_multi_filter.py
python lstm.py
python bidirectional_lstm.py
python cnn_lstm.py
python evaluation.py
