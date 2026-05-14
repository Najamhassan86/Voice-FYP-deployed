"""
Build learning_prototypes.json for the practice-score API.

Primary mode (when --dataset points to dataset.pkl):
  Loads training sequences, applies z-score normalization, runs the embedding
  submodel (penultimate layer), averages embeddings per class, L2-normalizes.

Fallback mode (no dataset or --use-classifier-weights):
  Uses L2-normalized columns of the final Dense softmax kernel (64 x num_classes)
  as reference directions in embedding space — reproducible without pickle data.

Usage (from repo root):
  .venv\\Scripts\\python transformer/models/training/build_learning_prototypes.py ^
    --model-dir transformer/models/saved_models/tcn_transformer_20260417_011511
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

# Default: this file is at transformer/models/training/build_learning_prototypes.py
REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_DIR = Path(__file__).resolve().parent


def _ensure_training_on_path() -> None:
    d = str(TRAINING_DIR)
    if d not in sys.path:
        sys.path.insert(0, d)


def load_full_model(model_dir: Path):
    _ensure_training_on_path()
    import tensorflow as tf
    from model_architecture import TCNBlock, TransformerBlock, PositionalEncoding

    model_path = model_dir / "best_model.h5"
    custom_objects = {
        "TCNBlock": TCNBlock,
        "TransformerBlock": TransformerBlock,
        "PositionalEncoding": PositionalEncoding,
    }
    model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects, compile=False)
    return model


def _pre_softmax_dense_output(model):
    """Last Dense layer before the final softmax (skips trailing Dropout layers)."""
    import tensorflow as tf

    layers = model.layers
    out_i = len(layers) - 1
    out = layers[out_i]
    out_cfg = out.get_config()
    if out_cfg.get("activation") != "softmax":
        raise ValueError(
            f"Expected last layer softmax, got activation={out_cfg.get('activation')!r}, name={out.name!r}"
        )
    i = out_i - 1
    while i >= 0 and isinstance(layers[i], tf.keras.layers.Dropout):
        i -= 1
    if i < 0 or not isinstance(layers[i], tf.keras.layers.Dense):
        raise ValueError("Could not locate pre-softmax Dense layer for embeddings")
    pen = layers[i]
    emb_dim = int(pen.units)
    return pen.output, emb_dim


def build_embedding_model(model):
    import tensorflow as tf

    pen_out, emb_dim = _pre_softmax_dense_output(model)
    emb_model = tf.keras.Model(inputs=model.input, outputs=pen_out, name="embedding_extractor")
    return emb_model, emb_dim


def load_norm_params(model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(model_dir / "normalization_params.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    mean = np.asarray(data["mean"], dtype=np.float32)
    std = np.asarray(data["std"], dtype=np.float32)
    std_safe = np.where(std == 0, 1.0, std)
    return mean, std_safe


def load_class_names(model_dir: Path) -> list[str]:
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        meta_path = REPO_ROOT / "transformer" / "data" / "extracted_landmarks" / "metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    names = meta.get("class_names", meta.get("classes", []))
    if not names:
        raise ValueError("No class_names in metadata")
    return list(names)


def normalize_sequences(x: np.ndarray, mean: np.ndarray, std_safe: np.ndarray) -> np.ndarray:
    """x: (N, 60, 188)"""
    return (x.astype(np.float32) - mean) / std_safe


def prototypes_from_dataset(
    emb_model, sequences: np.ndarray, labels: np.ndarray, class_names: list[str], mean: np.ndarray, std_safe: np.ndarray
) -> dict[str, list[float]]:
    import tensorflow as tf

    x = normalize_sequences(sequences, mean, std_safe)
    emb = emb_model.predict(x, verbose=0)
    num_classes = len(class_names)
    out: dict[str, list[float]] = {}
    for c in range(num_classes):
        mask = labels == c
        if not np.any(mask):
            vec = np.zeros(emb.shape[1], dtype=np.float32)
        else:
            vec = emb[mask].mean(axis=0)
        n = float(np.linalg.norm(vec))
        if n > 1e-8:
            vec = vec / n
        out[class_names[c]] = vec.astype(float).tolist()
    return out


def prototypes_from_classifier_weights(model, class_names: list[str]) -> dict[str, list[float]]:
    """Last Dense kernel shape (64, num_classes); column k is direction for class k."""
    last = model.layers[-1]
    w = last.get_weights()[0]
    if w.ndim != 2:
        raise ValueError(f"Unexpected weight shape {w.shape}")
    # (input_dim, units)
    num_classes = w.shape[1]
    if num_classes != len(class_names):
        raise ValueError(f"Kernel units {num_classes} != len(class_names) {len(class_names)}")
    out: dict[str, list[float]] = {}
    for k in range(num_classes):
        col = w[:, k].astype(np.float32)
        n = float(np.linalg.norm(col))
        if n > 1e-8:
            col = col / n
        out[class_names[k]] = col.astype(float).tolist()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / "transformer" / "models" / "saved_models" / "tcn_transformer_20260417_011511",
        help="Directory containing best_model.h5, normalization_params.json, metadata.json",
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Optional path to dataset.pkl")
    parser.add_argument(
        "--use-classifier-weights",
        action="store_true",
        help="Force classifier-weight prototypes (ignore dataset even if provided)",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    if not (model_dir / "best_model.h5").exists():
        raise SystemExit(f"Missing best_model.h5 in {model_dir}")

    class_names = load_class_names(model_dir)
    mean, std_safe = load_norm_params(model_dir)
    model = load_full_model(model_dir)
    emb_model, emb_dim = build_embedding_model(model)

    use_weights = args.use_classifier_weights or args.dataset is None
    prototype_source = "classifier_weights"
    prototypes: dict[str, list[float]]

    if not use_weights and args.dataset is not None and args.dataset.exists():
        with open(args.dataset, "rb") as f:
            ds = pickle.load(f)
        seq = np.asarray(ds["sequences"])
        labels = np.asarray(ds["labels"])
        prototypes = prototypes_from_dataset(emb_model, seq, labels, class_names, mean, std_safe)
        prototype_source = "training_mean_embedding"
    else:
        if args.dataset and not args.dataset.exists():
            print(f"Warning: dataset not found at {args.dataset}, using classifier weights.", file=sys.stderr)
        prototypes = prototypes_from_classifier_weights(model, class_names)

    payload = {
        "embedding_dim": emb_dim,
        "prototype_source": prototype_source,
        "class_names": class_names,
        "prototypes": prototypes,
    }

    out_path = model_dir / "learning_prototypes.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path} ({prototype_source}, dim={emb_dim}, classes={len(class_names)})")


if __name__ == "__main__":
    main()
