# from parex import ParExDataLoader, ParExTokenizer
# import torch.nn as nn
# import torch

# dataloader = ParExDataLoader('facebook/bart-base', 'sentence-transformers/all-MiniLM-L6-v2', 80, 40)
# [train_dataloader] = dataloader.get_dataloader(batch_size=2, types=['train'])
# for batch in train_dataloader:
#     break
# # print(batch)
# from parex import ParExModel
# model = ParExModel(
#     'sentence-transformers/all-MiniLM-L6-v2',
#     'parex_model/v1/gen',
#     load_pretrained_mapping=True,
#     mapping_ck='parex_model/v1/map')
# logits = model(batch)
# labels = batch.pop('decoder_labels')
# tokenizer = ParExTokenizer('facebook/bart-base')
# preds = [pred.strip() for pred in tokenizer.batch_decode(torch.argmax(logits, dim=-1), skip_special_tokens=True)]
# targets = [[sent.strip()] for sent in tokenizer.batch_decode(labels, skip_special_tokens=True)]
# print(preds)
# print(targets)
# # # from parex import ParExForConditionalGeneration
# # # gen = ParExForConditionalGeneration(model)
# # # print(gen.generate(batch))
# # from parex import ParExTokenizer
# # tokenizer = ParExTokenizer('facebook/bart-base')
# # # print(tokenizer.batch_decode([tokenizer("I<k>went to school").input_ids], skip_special_tokens=True))
# # tokenizer.save_pretrained('parex_tokenizer')

# Testing extractor
from ParEx import ParExExtractor
from transformers import AutoTokenizer, BartTokenizer, BartConfig
ckpt = 'sentence-transformers/all-MiniLM-L6-v2'
extractor = ParExExtractor(ckpt)
doc = """
         Supervised learning is the machine learning task of learning a function that
         maps an input to an output based on example input-output pairs. It infers a
         function from labeled training data consisting of a set of training examples.
         In supervised learning, each example is a pair consisting of an input object
         (typically a vector) and a desired output value (also called the supervisory signal).
         A supervised learning algorithm analyzes the training data and produces an inferred function,
         which can be used for mapping new examples. An optimal scenario will allow for the
         algorithm to correctly determine the class labels for unseen instances. This requires
         the learning algorithm to generalize from the training data to unseen situations in a
         'reasonable' way (see inductive bias).
      """
# keys = extractor.extract_keywords(doc)
# print(keys)
tokenizer = AutoTokenizer.from_pretrained(ckpt)
inputs = tokenizer(doc, padding='max_length', truncation=True, max_length=100, return_tensors='pt')
keyword_attention_mask = inputs.attention_mask
embed = extractor(**inputs)
# # print(embed)
# # for k, v in embed.items():
# #     print(f'{k}: {v.shape}')

# # Test generator
from ParEx import ParExModelGenerator, ParExGenerator
import torch

config = BartConfig.from_pretrained('new_bart')
config.update({'keyword_output_dim': 384, 'generate_input_dim' : 768})
model = ParExGenerator(config)
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
inputs = bart_tokenizer(doc, padding='max_length', truncation=True, max_length=20, return_tensors='pt')
# output = model(**inputs, encoder_keyword_embeds=embed.last_hidden_state, encoder_keyword_attention_mask=keyword_attention_mask)
# print(output)
# # for k, v in output.items():
# #     if v is not None:
# #         print(f'{k}: {v.shape}')

# preds_tgt = self.model.generate(batch['input_ids'], max_length=128, num_beams=4)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, LogitsProcessorList, MinLengthLogitsProcessor, BeamSearchScorer, StoppingCriteriaList, MaxLengthCriteria
num_beams = 4
input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
input_ids = input_ids * model.config.decoder_start_token_id
print(f'Encoder input ids shape: {inputs.input_ids.shape}')
print(f'Embed last hidden state: {embed.last_hidden_state.shape}')
encoder_ouputs = model.get_encoder()(inputs.input_ids.repeat_interleave(num_beams, dim=0), return_dict=True)
with torch.no_grad():
      upsize_embed = model.mappings(embed.last_hidden_state.repeat_interleave(num_beams, dim=0))
      encoder_ouputs.last_hidden_state = torch.cat([encoder_ouputs.last_hidden_state, upsize_embed], dim=1)
model_kwargs = {
      "encoder_outputs": encoder_ouputs
}
beam_scorer = BeamSearchScorer(
      batch_size=1,
      num_beams=num_beams,
      device=model.device,
)
logits_processor = LogitsProcessorList([
      MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
])
stopping_criteria=StoppingCriteriaList([
      MaxLengthCriteria(max_length=128)
])

outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, stopping_criteria=stopping_criteria, **model_kwargs)
print(outputs)
print(bart_tokenizer.batch_decode(outputs, skip_special_tokens=True))

# from ParEx import ParExDataLoader, ParExModel
# from transformers import BartConfig
# parex_dataloader = ParExDataLoader('facebook/bart-base', 'sentence-transformers/all-MiniLM-L6-v2', 80, 40)
# [train_dataloader] = parex_dataloader.get_dataloader(batch_size=2, types=['train'])
# for batch in train_dataloader:
#       break
# # print(batch)
# config = BartConfig.from_pretrained('new_bart')
# config.update({'keyword_output_dim': 384, 'generate_input_dim' : 768})
# model = ParExModel('sentence-transformers/all-MiniLM-L6-v2', config)
# # output = model(batch)
# # print(output)
# out = model.generate(batch, num_beams=4, max_length=10)
# print(out.shape)