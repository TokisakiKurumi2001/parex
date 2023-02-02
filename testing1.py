from ParEx import ParExDataLoader, ParExModel, ParExConfig
from transformers import BartConfig
parex_dataloader = ParExDataLoader('facebook/bart-base', 'sentence-transformers/all-MiniLM-L6-v2', 80, 40)
[train_dataloader] = parex_dataloader.get_dataloader(batch_size=2, types=['train'])
for batch in train_dataloader:
      break
# print(batch)
model = ParExModel('sentence-transformers/all-MiniLM-L6-v2', 'new_bart')
# output = model(batch)
# print(output)
out = model.generate(batch, num_beams=4, max_length=10)
print(out.shape)
# model.save_pretrained('testing')
# model1 = ParExModel('sentence-transformers/all-MiniLM-L6-v2', None, 'eval', 'testing')
# out = model.generate(batch, num_beams=4, max_length=10)
# print(out.shape)