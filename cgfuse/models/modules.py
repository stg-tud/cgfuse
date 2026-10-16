import torch
import torch.nn as nn

from transformers import (
    AutoModel,
    EncoderDecoderConfig,
    EncoderDecoderModel,
)


class RobertaClassificationHead(nn.Module):
    """
    Head used for Encoder -> EncoderDecoder model in generation.
    From: https://github.com/NougatCA/FineTuner/blob/main/src/models.py
    Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

def get_lm_head(model):
    """Get the language modeling head from different model types"""
    if hasattr(model, 'lm_head'):
        return model.lm_head
    elif hasattr(model, 'cls') and hasattr(model.cls, 'predictions'):
        return model.cls.predictions.decoder  # BERT
    elif hasattr(model, 'score'):
        return model.score  # Some models
    elif hasattr(model, 'output_projection'):
        return model.output_projection
    else:
        return None

def set_lm_head(model, new_head):
    """Set the language modeling head for different model types"""
    if hasattr(model, 'lm_head'):
        model.lm_head = new_head
    elif hasattr(model, 'cls') and hasattr(model.cls, 'predictions'):
        model.cls.predictions.decoder = new_head
    elif hasattr(model, 'score'):
        model.score = new_head
    elif hasattr(model, 'output_projection'):
        model.output_projection = new_head

def ED_from_encoder_model(hf_handle):
    # From: https://github.com/NougatCA/FineTuner/blob/main/src/models.py
    """Plus: EncoderDecoder model with shared embeddings like T5"""

    # Load the base encoder model
    encoder = AutoModel.from_pretrained(hf_handle)

    # Create encoder-decoder config
    encoder_config = encoder.config
    decoder_config = encoder_config  # Use same config for decoder

    seq2seq_config = EncoderDecoderConfig.from_encoder_decoder_configs(
        encoder_config, decoder_config
    )

    # Create the model
    model = EncoderDecoderModel(config=seq2seq_config, encoder=encoder)

    # Get the original encoder embeddings
    original_embeddings = encoder.get_input_embeddings()

    # Create a shared embedding layer (same as original)
    shared_embeddings = nn.Embedding(
        original_embeddings.num_embeddings,
        original_embeddings.embedding_dim,
        padding_idx=original_embeddings.padding_idx
    )

    # Copy weights from original embeddings
    shared_embeddings.weight.data.copy_(original_embeddings.weight.data)

    # Set shared embeddings for both encoder and decoder
    model.encoder.set_input_embeddings(shared_embeddings)
    model.decoder.set_input_embeddings(shared_embeddings)

    # Store reference to shared embeddings (T5-style)
    model.shared = shared_embeddings

    config_decoder = seq2seq_config.decoder
    config_decoder.is_decoder = True
    config_decoder.add_cross_attention = True
    # [Naugat] End model initialization

    return model
