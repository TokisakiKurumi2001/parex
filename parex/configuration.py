from transformers import PretrainedConfig

class ParExMappingConfig(PretrainedConfig):
    model_type = "ParEx_Mapping"
    def __init__(
        self, hidden_dim1: int=384, hidden_dim2: int=768, 
        pad_token_id: int=1, eos_token_id: int=3, is_encoder_decoder: bool=False,
        decoder_start_token_id: int=2, forced_eos_token_id: int=3,
        **kwargs):
        super(ParExMappingConfig, self).__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs
        )
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2