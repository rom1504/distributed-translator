"""
This script show how we can do translation using multiple processes

Usage:
python translation_speed.py model_name
"""
import os
from easynmt import util, EasyNMT
import gzip
import csv
import sys
import time
import logging
import pyarrow.parquet  as pq
import pandas as pd


if __name__ == '__main__':
    model = EasyNMT('opus-mt')

    df = pd.read_parquet("/home/rom1504/metadata_0000.parquet")
    sentences = df["caption"].tolist()
    sentences = sentences[:10000]
    print(sentences)

    ######## Multi-Process-Translation
    # You can pass a target_devices parameter to the start_multi_process_pool() method to define how many processes to start
    # and on which devices the processes should run
    process_pool = model.start_multi_process_pool(['cpu'])

    #Do some warm-up
    model.translate_multi_process(process_pool, sentences[0:100], source_lang='en', target_lang='de', show_progress_bar=False, batch_size=12)

    # Start translation speed measure - Multi process
    start_time = time.time()
    translations_multi_p = model.translate_multi_process(process_pool, sentences, source_lang='en', target_lang='de', show_progress_bar=True, batch_size=12)
    end_time = time.time()
    print("Multi-Process translation done after {:.2f} sec. {:.2f} sentences / second".format(end_time - start_time, len(sentences) / (end_time - start_time)))


    model.stop_multi_process_pool(process_pool)