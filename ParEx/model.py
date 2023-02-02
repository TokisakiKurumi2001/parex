from transformers import BertModel, PreTrainedModel, LogitsProcessorList, MinLengthLogitsProcessor, BeamSearchScorer, StoppingCriteriaList, MaxLengthCriteria
from keybert import KeyBERT
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput
from ParEx import ParExConfig
from typing import Union, List, Optional, Tuple
from torch import Tensor
import torch
import torch.nn as nn

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def pad_sequence(tensor_input: Tensor, max_length: int, padding_value: int) -> Tensor:
    # tensor_input: (1, seq_len)
    curr_len = tensor_input.shape[1]
    if curr_len < max_length:
        pad_len = max_length - curr_len
        return torch.cat([tensor_input, torch.full((1, pad_len), padding_value, device=tensor_input.device)], dim=1)
    else: 
        return tensor_input

class ParExExtractor(nn.Module):
    def __init__(self, ckpt):
        super(ParExExtractor, self).__init__()
        self.model = BertModel.from_pretrained(ckpt)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        # self.key_extractor = KeyBERT(model=self.model)

    # def extract_keywords(self, sent: str) -> List[str]:
    #     keywords = [k[0] for k in self.key_extractor.extract_keywords(sent, keyphrase_ngram_range=(1, 1), stop_words='english')]
    #     return keywords

    def forward(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor):
        with torch.no_grad():
            embeddings = self.model(input_ids, attention_mask)
        return embeddings

class ParExPretrainedModel(PreTrainedModel):
    config_class = ParExConfig
    base_model_prefix = "parex"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = [r"encoder.version", r"decoder.version"]
    _no_split_modules = [r"BartEncoderLayer", r"BartDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (BartDecoder, BartEncoder)):
            module.gradient_checkpointing = value

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs

class ParExModelGenerator(ParExPretrainedModel):
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: ParExConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.mappings = nn.Linear(config.keyword_output_dim, config.generate_input_dim)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        encoder_keyword_embeds: torch.Tensor = None,
        encoder_keyword_attention_mask: torch.Tensor = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            attention_mask = torch.cat([attention_mask, encoder_keyword_attention_mask], dim=1)
            upsize_embed = self.mappings(encoder_keyword_embeds)
            encoder_outputs.last_hidden_state = torch.cat([encoder_outputs.last_hidden_state, upsize_embed], dim=1)
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class ParExGenerator(ParExPretrainedModel):
    base_model_prefix = "parex"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"lm_head.weight",
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    def __init__(self, config: ParExConfig):
        super().__init__(config)
        self.model = ParExModelGenerator(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        encoder_keyword_embeds: torch.FloatTensor = None,
        encoder_keyword_attention_mask: torch.LongTensor = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_keyword_embeds=encoder_keyword_embeds,
            encoder_keyword_attention_mask=encoder_keyword_attention_mask,
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        return Seq2SeqLMOutput(
            loss=None,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

class ParExModel(nn.Module):
    def __init__(self, extractor_ckpt: str, generator_ckpt: str, mode: str='train', pretrained_ckpt: str=''):
        super(ParExModel, self).__init__()
        self.extractor = ParExExtractor(extractor_ckpt)
        if mode == "train":
            self.generator = ParExGenerator.from_pretrained(generator_ckpt)
        else:
            self.generator = ParExGenerator.from_pretrained(pretrained_ckpt)

    def save_pretrained(self, path):
        self.generator.config.model_type = "parex"
        self.generator.save_pretrained(path)

    def forward(self, inputs):
        keyword_embeds = self.extractor(**inputs['extract'])

        outputs = self.generator(**inputs['keyword'], decoder_input_ids=inputs['decoder_input_ids'], encoder_keyword_embeds=keyword_embeds.last_hidden_state, encoder_keyword_attention_mask=inputs['extract'].attention_mask)
        return outputs

    def generate(self, batch, num_beams, max_length):
        batch_size = batch['keyword'].input_ids.shape[0]
        device = self.generator.device
        decoder_start_token_id = self.generator.config.decoder_start_token_id
        eos_token_id = self.generator.config.eos_token_id
        pad_token_id = self.generator.config.pad_token_id
        # print(f'Device of extractor: {self.extractor.model.device}')
        # print(f"Device of batch: {batch['extract'].input_ids.device}")
        keyword_embeds = self.extractor(**batch['extract'])
        outputs = []
        num_beams = num_beams
        for i in range(0, batch_size):
            encoder_input_ids = batch['keyword'].input_ids[i].unsqueeze(0)
            embed = keyword_embeds.last_hidden_state[i].unsqueeze(0)
            # print(encoder_input_ids.shape)
            # print(embed.shape)
      
            input_ids = torch.ones((num_beams, 1), device=device, dtype=torch.long)
            input_ids = input_ids * decoder_start_token_id
            encoder_ouputs = self.generator.get_encoder()(encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True)
            # print(encoder_ouputs.last_hidden_state.shape)
            with torch.no_grad():
                upsize_embed = self.generator.model.mappings(embed.repeat_interleave(num_beams, dim=0))
                # print(upsize_embed.shape)
                encoder_ouputs.last_hidden_state = torch.cat([encoder_ouputs.last_hidden_state, upsize_embed], dim=1)
            model_kwargs = {
                "encoder_outputs": encoder_ouputs
            }
            beam_scorer = BeamSearchScorer(
                batch_size=1,
                num_beams=num_beams,
                device=device,
            )
            logits_processor = LogitsProcessorList([
                MinLengthLogitsProcessor(5, eos_token_id=eos_token_id),
            ])
            stopping_criteria=StoppingCriteriaList([
                MaxLengthCriteria(max_length=max_length)
            ])

            output = self.generator.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, stopping_criteria=stopping_criteria, **model_kwargs)
            output = pad_sequence(output, max_length, pad_token_id)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)
        return outputs

    
