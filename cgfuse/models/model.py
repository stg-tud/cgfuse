from torch import nn


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.embeddings: nn.Embedding = None  # to be defined in child class

    def resize_token_embeddings(self, new_num_tokens):
        old_num_tokens, old_embedding_dim = self.embeddings.weight.size()

        assert new_num_tokens > old_num_tokens, "New number of tokens must be greater than the old number."

        # Calculate mean and std of original embeddings for initialization
        original_mean = self.embeddings.weight.mean(dim=0)
        original_std = self.embeddings.weight.std(dim=0)

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(
            self.embeddings.weight.device, dtype=self.embeddings.weight.dtype
        )

        # Copy token embeddings from the previous weights
        n = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:n, :] = self.embeddings.weight.data[:n, :]

        # Initialize newly added embeddings with same mean and std as original
        if new_num_tokens > old_num_tokens:
            # Generate random weights with same statistical properties
            import torch

            new_weights = torch.randn(
                new_num_tokens - old_num_tokens,
                old_embedding_dim,
                device=self.embeddings.weight.device,
                dtype=self.embeddings.weight.dtype,
            )
            # Scale to match original std and shift to match original mean
            new_weights = new_weights * original_std + original_mean
            new_embeddings.weight.data[old_num_tokens:, :] = new_weights

        return new_embeddings
