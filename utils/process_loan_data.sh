#!/usr/bin/env bash
# Place lending-club-loan-data.zip in this directory before running.

unzip lending-club-loan-data.zip -d ./lending-club-loan-data
if [ ! -d ../data  ];then
  mkdir ../data
else
  echo '../data' dir exist
fi

mv ./lending-club-loan-data ../data/
echo move 'lending-club-loan-data' dir to '../data/lending-club-loan-data'
chmod a+r ../data/lending-club-loan-data/loan.csv
python loan_preprocess.py
