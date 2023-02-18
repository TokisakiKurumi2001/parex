from transformers import BartForConditionalGeneration
model = BartForConditionalGeneration.from_pretrained('new_bart')
model.push_to_hub('bart_init')
