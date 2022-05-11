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
import math

if __name__ == '__main__':
    model = EasyNMT('m2m_100_418M')
    #model = EasyNMT('opus-mt')
    langs = "af, am, ar, ast, az, ba, be, bg, bn, br, bs, ca, ceb, cs, cy, da, de, el, en, es, et, fa, ff, fi, fr, fy, ga, gd, gl, gu, ha, he, hi, hr, ht, hu, hy, id, ig, ilo, is, it, ja, jv, ka, kk, km, kn, ko, lb, lg, ln, lo, lt, lv, mg, mk, ml, mn, mr, ms, my, ne, nl, no, ns, oc, or, pa, pl, ps, pt, ro, ru, sd, si, sk, sl, so, sq, sr, ss, su, sv, sw, ta, th, tl, tn, tr, uk, ur, uz, vi, wo, xh, yi, yo, zh, zu".split(", ")
    for i in range(8):
        df = pd.read_parquet(f"https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/laion2B-en-metadata/metadata_{i:04}.parquet")
        sentences = df["caption"].tolist()
        #sentences = sentences[:1000000]
        print(sentences)

        ######## Multi-Process-Translation
        # You can pass a target_devices parameter to the start_multi_process_pool() method to define how many processes to start
        # and on which devices the processes should run
        process_pool = model.start_multi_process_pool(['cuda:0', 'cuda:1', 'cuda:2','cuda:3','cuda:4', 'cuda:5','cuda:6', 'cuda:7'])

        #Do some warm-up
        model.translate_multi_process(process_pool, sentences[0:100], source_lang='en', target_lang='de', show_progress_bar=False, batch_size=32)

        # Start translation speed measure - Multi process
        start_time = time.time()
        lang_chunk_size=math.ceil(len(sentences) / len(langs))
        results=[]
        ls = []
        for i, l in enumerate(langs):
            chunk = sentences[i*lang_chunk_size:(i+1)*lang_chunk_size]
            translations_multi_p = model.translate_multi_process(process_pool, chunk, source_lang='en', target_lang=l, show_progress_bar=True, batch_size=32)
            results.extend(translations_multi_p)
            ls.extend([l]*len(translations_multi_p))
        
        df["translation"]=results
        df["translation_language"]=ls
        df.to_parquet(f"output_{i:04}.parquet")
        end_time = time.time()
        print("Multi-Process translation done after {:.2f} sec. {:.2f} sentences / second".format(end_time - start_time, len(sentences) / (end_time - start_time)))


        model.stop_multi_process_pool(process_pool)