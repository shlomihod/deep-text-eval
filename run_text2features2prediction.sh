#!/bin/bash

set -x
set -e
shopt -s expand_aliases

export CUDA_VISIBLE_DEVICES=0
cd ~/deep-text-eval/text2features/
python3 nlp_extractor.py ../data/weebit/weebit.h5

alias nbx='jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=-1'

cd ~/deep-text-eval/text2features/
nbx features_analysis.ipynb

cd ~/deep-text-eval/features2prediction/weebit/
nbx classification.ipynb
nbx regression.ipynb
nbx ordinal-regression.ipynb

nbx models-comparison.ipynb

cd ~/deep-text-eval/
