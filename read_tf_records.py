import tensorflow as tf
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./data/pretrain_tfrecords/tf_examples.tfrecord")
    parser.add_argument("--output", type=str, default="data/pretrain_tfrecords/read_tf_records.txt")
    return parser.parse_args()

args = parse_args()

raw_dataset = tf.data.TFRecordDataset(args.input)

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    f = open(args.output, "w")
    f.write(str(example))
