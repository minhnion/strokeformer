#! /bin/bash

while read DATA_SET; do
   while read MODEL_TYPE; do
        echo "Running model $MODEL_TYPE for data set $DATA_SET"
        python main.py --train_mode "$MODEL_TYPE" --downstream_name "$DATA_SET" > "output/output_${MODEL_TYPE}_${DATA_SET}.txt"
   done < model_types.txt
done < data1_types.txt

