from parex import ParExForConditionalGeneration, ParExModel
from parex import ParExDataLoader
from transformers import AutoTokenizer

dataloader = ParExDataLoader('facebook/bart-base', 'sentence-transformers/all-MiniLM-L6-v2', 80, 40)
[train_dataloader] = dataloader.get_dataloader(batch_size=1, types=['train'])
from parex import ParExModel
model = ParExModel(
    'sentence-transformers/all-MiniLM-L6-v2',
    'parex_model/v1/gen',
    load_pretrained_mapping=True,
    mapping_ck='parex_model/v1/map')
gen = ParExForConditionalGeneration(model)
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
limit = 10
for i, batch in enumerate(train_dataloader):
    out = gen.generate(batch)[0]
    print(tokenizer.decode(out, skip_special_tokens=True))
    if i > limit:
        break