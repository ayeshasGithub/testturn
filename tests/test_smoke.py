import torch
from toxicity_transformer.models.transformer import ModelConfig, ToxicityTransformer

def test_forward_smoke():
    cfg = ModelConfig(vocab_size=100, max_len=16, d_model=32, n_heads=4, n_layers=2)
    model = ToxicityTransformer(cfg, pad_id=0, cls_id=2)
    input_ids = torch.randint(0, 100, (4, 16))
    attention_mask = torch.ones((4, 16), dtype=torch.long)
    logits = model(input_ids, attention_mask)
    assert logits.shape == (4,)
