#!/bin/bash


usage ()
{
  echo 'Usage : Script <corpus name>'
  exit
}

if [ "$#" -ne 1 ]
then
  usage
fi


set -x
set -e
shopt -s expand_aliases

alias nbx='jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=-1'

export CUDA_VISIBLE_DEVICES=0

cd ~/deep-text-eval/text2features
python3 nlp_extractor.py "../data/$1/$1.h5"

cd ~/deep-text-eval/text2features/$1
nbx features_analysis.ipynb

cd ~/deep-text-eval/features2prediction/$1
nbx classification.ipynb
nbx regression.ipynb
nbx ordinal-regression.ipynb

nbx models-comparison.ipynb
nbx features-selection.ipynb

cd ~/deep-text-eval/
