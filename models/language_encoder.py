import torch
import torch.nn as nn

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None


class LanguageEncoder(nn.Module):
    def __init__(
        self,
        model_name="distilbert-base-uncased",
        freeze=True,
        cache_dir=".cache/huggingface",
        hidden_dim=768,
        vocab_size=4096,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_transformer = False

        if AutoTokenizer is not None and AutoModel is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    local_files_only=True,
                )
                self.model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    local_files_only=True,
                )
                self.hidden_dim = self.model.config.hidden_size
                self.use_transformer = True

                if freeze:
                    for param in self.model.parameters():
                        param.requires_grad = False
            except Exception:
                self.tokenizer = None
                self.model = None

        if not self.use_transformer:
            self.embedding = nn.EmbeddingBag(vocab_size, hidden_dim, mode="mean")
            self.vocab_size = vocab_size

    def _hash_tokenize(self, text_list, device):
        token_ids = []
        offsets = [0]

        for text in text_list:
            tokens = str(text).lower().split()
            ids = [hash(token) % self.vocab_size for token in tokens]
            if not ids:
                ids = [0]

            token_ids.extend(ids)
            offsets.append(len(token_ids))

        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
        offset_tensor = torch.tensor(offsets[:-1], dtype=torch.long, device=device)
        return token_tensor, offset_tensor

    def forward(self, text_list):
        if self.use_transformer:
            inputs = self.tokenizer(
                text_list,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            device = next(self.model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0]

        device = self.embedding.weight.device
        token_ids, offsets = self._hash_tokenize(text_list, device)
        return self.embedding(token_ids, offsets)
