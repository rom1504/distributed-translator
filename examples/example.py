from distributed_translator import hello_world

import pyarrow.parquet  as pq
from tqdm import tqdm

hello_world("hi")

from easynmt import EasyNMT
model = EasyNMT('opus-mt')

#Translate a single sentence to German
print(model.translate('This is a sentence we want to translate to German', target_lang='fr'))


#df = pq.read_parquet("/home/rom1504/metadata_0000.parquet")

import pyarrow.dataset as ds
from tqdm import tqdm

model.translate_stream


dataset = ds.dataset("/home/rom1504/metadata_0000.parquet")

for batch in tqdm(dataset.to_batches(columns=["caption"], filter=~ds.field("caption").is_null(), batch_size=32)):
    captions = batch.column("caption").to_pylist()
    print(captions)
    print(model.translate_sentences(captions, source_lang="en", target_lang='fr', batch_size=32))