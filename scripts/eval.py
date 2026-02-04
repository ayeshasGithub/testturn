from __future__ import annotations

import argparse
import json
import torch
import numpy as np

from toxicity_transformer.data.dataset import DataConfig, load_civil_comments, add_binary_label, stratified_split, tokenize_dataset
from toxicity_transformer.data.collate import collate_fn
from toxicity_transformer.data.tokenizer import Vocab
from toxicity_transformer.models.transformer import ModelConfig, ToxicityTransformer
from toxicity_transformer.training.trainer import make_loader, evaluate

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--run-dir", type=str, default=None, help="Directory containing vocab.json; defaults to checkpoint's dir")
    p.add_argument("--max-examples", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--dataset-path", type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    run_dir = args.run_dir or str((__import__("pathlib").Path(args.checkpoint)).parent)

    with open(f"{run_dir}/vocab.json", "r", encoding="utf-8") as f:
        itos = json.load(f)["itos"]
    stoi = {t:i for i,t in enumerate(itos)}
    vocab = Vocab(stoi=stoi, itos=itos)

    # Load a subset and evaluate on test split
    dcfg = DataConfig(dataset_path=args.dataset_path, max_len=args.max_len)
    raw = load_civil_comments(dcfg, max_examples=args.max_examples)
    raw = add_binary_label(raw, dcfg)
    dsd = stratified_split(raw, dcfg)

    test_ds = tokenize_dataset(dsd["test"], dcfg, vocab, pretokenize_dir=None)

    collate = lambda feats: collate_fn(feats, pad_id=vocab.pad_id)
    test_loader, _ = make_loader(test_ds, collate, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild model from checkpoint shapes
    state = ckpt["model"]
    # Infer d_model from embedding weight
    d_model = state["tok_emb.weight"].shape[1]
    vocab_size = state["tok_emb.weight"].shape[0]
    # Infer layers by counting blocks
    n_layers = len({k.split(".")[1] for k in state.keys() if k.startswith("blocks.")})
    # heads not inferable robustly; store common default; can be overridden if needed
    mcfg = ModelConfig(vocab_size=vocab_size, max_len=args.max_len, d_model=d_model, n_heads=4, n_layers=n_layers)

    model = ToxicityTransformer(mcfg, pad_id=vocab.pad_id, cls_id=vocab.cls_id).to(device)
    model.load_state_dict(state, strict=False)

    metrics = evaluate(model, test_loader, device)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
