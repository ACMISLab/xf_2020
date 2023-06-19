#!/usr/bin/env sh

DATA_ROOT=$1

CURR_DIR=$(pwd)


cd $DATA_ROOT || exit



mkdir -p ./round2_data/Annotations
mkdir -p ./round2_data/defect
mkdir -p ./round2_data/normal
echo $(pwd)
cp ./guangdong1_round2_train2_20191004_Annotations/Annotations/anno_train.json \
./round2_data/Annotations/anno_train_1004.json
cp ./guangdong1_round2_train_part1_20190924/Annotations/anno_train.json ./round2_data/Annotations/anno_train_0924.json
cp -r  ./guangdong1_round2_train_part1_20190924/defect/* ./round2_data/defect/
cp -r  ./guangdong1_round2_train_part2_20190924/defect/* ./round2_data/defect/
cp -r  ./guangdong1_round2_train_part3_20190924/defect/* ./round2_data/defect/
cp -r  ./guangdong1_round2_train_part2_20190924/normal/* ./round2_data/normal/
cp -r  ./guangdong1_round2_train2_20191004_images/defect/* ./round2_data/defect/
cp -r  ./guangdong1_round2_train2_20191004_images/normal/* ./round2_data/normal/

cd $CURR_DIR || exit

python tools/fabric_convert.py $DATA_ROOT



