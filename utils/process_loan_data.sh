#!/usr/bin/env bash
# Place lending-club-loan-data.zip in this directory before running.

unzip lending-club-loan-data.zip -d ./lending-club-loan-data
data_dir="../data"
if [ ! -d "$data_dir"  ];then
  mkdir "$data_dir"
else
  echo "$data_dir dir exists"
fi

mv ./lending-club-loan-data "$data_dir"
echo "move lending-club-loan-data dir to $data_dir"
chmod a+r "$data_dir/lending-club-loan-data/loan.csv"
python loan_preprocess.py
