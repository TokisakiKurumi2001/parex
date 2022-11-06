from transformers import BertModel, BartForConditionalGeneration, PreTrainedModel
from parex import ParExMappingConfig
from typing import Union
from torch import Tensor
import torch
import torch.nn as nn

class ParExMappingPretrainedModel(PreTrainedModel):
    config_class = ParExMappingConfig
    base_model_prefix = "ParEx_Mapping"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: Union[nn.Linear, nn.Embedding]):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class ParExMappingModel(ParExMappingPretrainedModel):
    def __init__(self, config: ParExMappingConfig):
        super(ParExMappingModel, self).__init__(config)
        self.fc = nn.Linear(self.config.hidden_dim1, self.config.hidden_dim2)

    def forward(self, input):
        return self.fc(input)

class ParExModel(nn.Module):
    def __init__(
        self, extract_model_ck: str, reconstruct_model_ck: str,
        load_pretrained_mapping: bool, mapping_ck: str):
        super(ParExModel, self).__init__()
        
        self.extract_model = BertModel.from_pretrained(extract_model_ck)
        for p in self.extract_model.parameters():
            p.requires_grad = False
        
        if load_pretrained_mapping == False:
            self.reconstruct_model = BartForConditionalGeneration.from_pretrained(reconstruct_model_ck)
            self.reconstruct_model.resize_token_embeddings(50265 + 2)
            mapping_config = ParExMappingConfig()
            self.mapping = ParExMappingModel(mapping_config)
        else:
            self.reconstruct_model = BartForConditionalGeneration.from_pretrained(reconstruct_model_ck)
            self.mapping = ParExMappingModel.from_pretrained(mapping_ck)

    def save_pretrained(self, path):
        self.reconstruct_model.save_pretrained(path)

    def save_mapping(self, path):
        self.mapping.save_pretrained(path)

    def forward(self, input):
        input_for_embed = input['extract']
        keyword = input['keyword']
        embed = self.extract_model(**input_for_embed).last_hidden_state
        upsize_embed = self.mapping(embed)
        out_enc = self.reconstruct_model.get_encoder()(**keyword).last_hidden_state
        new_hidden_state = torch.cat([out_enc, upsize_embed], dim=1)
        # feed into decoder
        out = self.reconstruct_model.get_decoder()(
            input_ids=input['decoder_input_ids'],
            encoder_hidden_states=new_hidden_state
        ).last_hidden_state
        lm_logits = self.reconstruct_model.lm_head(out) + self.reconstruct_model.final_logits_bias
        return lm_logits

class ParExForConditionalGeneration:
    def __init__(self, model: ParExModel, alpha: float=0.6):
        self.model = model
        self.alpha = alpha

    def __sequence_length_penalty(self, length: int, alpha: float) -> float:
            """ Sequence length penalty for beam search.
            
            Source: Google's Neural Machine Translation System (https://arxiv.org/abs/1609.08144)
            """
            return ((5 + length) / (5 + 1)) ** alpha

    def generate(self, input, num_beam: int=4, max_output_length: int=80, vocab_size: int=50267) -> Tensor:
        self.model.eval()
        with torch.no_grad():
            input_for_embed = input['extract']
            keyword = input['keyword']
            embed = self.model.extract_model(**input_for_embed).last_hidden_state
            upsize_embed = self.model.mapping(embed)
            out_enc = self.model.reconstruct_model.get_encoder()(**keyword).last_hidden_state
            new_hidden_state = torch.cat([out_enc, upsize_embed], dim=1).squeeze(0)

            return self.__decode(new_hidden_state, max_output_length, vocab_size, num_beam)

    def __decode(
        self, encoder_output: Tensor, max_output_length: int, vocab_size: int,
        num_beam: int, bos_token_id: int=2, eos_token_id: int=2
    ):
        # Start with <bos>
        decoder_input = torch.Tensor([[bos_token_id]]).long()
        scores = torch.Tensor([0.])
        for i in range(max_output_length):
            # Encoder output expansion from the second time step to the beam size
            if i==1:
                encoder_output = encoder_output.expand(num_beam, *encoder_output.shape[1:])

            # prediction
            out = self.model.reconstruct_model.get_decoder()(
                input_ids=decoder_input,
                encoder_hidden_states=encoder_output
            ).last_hidden_state
            logits = self.model.reconstruct_model.lm_head(out) + self.model.reconstruct_model.final_logits_bias
            logits = logits[:, -1] # Last sequence step: [beam_size, sequence_length, vocab_size] => [beam_size, vocab_size]

            # Softmax
            log_probs = torch.log_softmax(logits, dim=1)
            log_probs = log_probs / self.__sequence_length_penalty(i+1, self.alpha)

            # Update score where EOS has not been reched
            log_probs[decoder_input[:, -1]==eos_token_id, :] = 0
            scores = scores.unsqueeze(1) + log_probs # scores [beam_size, 1], log_probs [beam_size, vocab_size]

            # Flatten scores from [beams, vocab_size] to [beams * vocab_size] to get top k, and reconstruct beam indices and token indices
            scores, indices = torch.topk(scores.reshape(-1), num_beam)
            beam_indices  = torch.divide   (indices, vocab_size, rounding_mode='floor') # indices // vocab_size
            token_indices = torch.remainder(indices, vocab_size)                        # indices %  vocab_size

            # Build the next decoder input
            next_decoder_input = []
            for beam_index, token_index in zip(beam_indices, token_indices):
                prev_decoder_input = decoder_input[beam_index]
                if prev_decoder_input[-1]==eos_token_id:
                    token_index = eos_token_id # once EOS, always EOS
                token_index = torch.LongTensor([token_index])
                next_decoder_input.append(torch.cat([prev_decoder_input, token_index]))
            decoder_input = torch.vstack(next_decoder_input)

            # If all beams are finished, exit
            if (decoder_input[:, -1]==eos_token_id).sum() == num_beam:
                break

        # convert the top scored sequence to a list of text tokens
        decoder_output, _ = max(zip(decoder_input, scores), key=lambda x: x[1])
        decoder_output = decoder_output[1:] # remove SOS
        return decoder_input
