"""
PSL (Pakistan Sign Language) Inference Service

Loads the trained TCN + Transformer model and provides inference functionality.
Model expects input shape (60, 188) - 60 frames of 188-dimensional feature vectors.
"""

import os
import sys
import json
import shutil
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy import TensorFlow (heavy dependency)
_model = None
_embedding_model = None
_normalization_params = None
_class_labels = None
_model_loaded = False
_custom_objects = None
_model_load_error = None
# Rows aligned with _class_labels: (num_classes, embedding_dim), L2-normalized
_prototype_matrix: Optional[np.ndarray] = None
_prototype_source: Optional[str] = None
_practice_scoring_ready = False


def _get_model_paths():
    """Resolve model-related paths by scanning supported project folders."""
    base_dir = Path(__file__).parent.parent.parent.parent
    model_dir_override = os.getenv("MODEL_DIR", "").strip()

    if model_dir_override:
        override_dir = Path(model_dir_override)
        if not override_dir.is_absolute():
            override_dir = base_dir / override_dir

        metadata_override = override_dir / "metadata.json"
        if not metadata_override.exists():
            metadata_override = base_dir / "transformer" / "data" / "extracted_landmarks" / "metadata.json"

        return {
            "model": override_dir / "best_model.h5",
            "normalization": override_dir / "normalization_params.json",
            "metadata": metadata_override,
            "training_module": base_dir / "transformer" / "models" / "training",
        }

    # Candidate roots in priority order.
    # 1) transformer/ is the active training pipeline used by backend docs.
    # 2) TransformerWala/ is an alternate location present in this workspace.
    root_candidates = [
        base_dir / "transformer",
        base_dir / "TransformerWala",
    ]

    for root in root_candidates:
        saved_models = root / "models" / "saved_models"
        metadata_file = root / "data" / "extracted_landmarks" / "metadata.json"
        training_module = root / "models" / "training"

        if not saved_models.exists() or not training_module.exists():
            continue

        # Pick the most recently modified directory that has both model + normalization files.
        candidates = sorted(
            [p for p in saved_models.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for model_dir in candidates:
            model_file = model_dir / "best_model.h5"
            norm_file = model_dir / "normalization_params.json"
            if model_file.exists() and norm_file.exists():
                # Prefer metadata produced with this exact model artifact.
                resolved_metadata = model_dir / "metadata.json"
                if not resolved_metadata.exists():
                    resolved_metadata = metadata_file

                logger.info(f"Using PSL model directory: {model_dir}")
                return {
                    "model": model_file,
                    "normalization": norm_file,
                    "metadata": resolved_metadata,
                    "training_module": training_module,
                }

    # Explicit fallback path to preserve previous behavior if no candidate matched.
    fallback_model_dir = base_dir / "transformer" / "models" / "saved_models" / "tcn_transformer_20260417_011511"
    fallback_metadata = fallback_model_dir / "metadata.json"
    if not fallback_metadata.exists():
        fallback_metadata = base_dir / "transformer" / "data" / "extracted_landmarks" / "metadata.json"

    return {
        "model": fallback_model_dir / "best_model.h5",
        "normalization": fallback_model_dir / "normalization_params.json",
        "metadata": fallback_metadata,
        "training_module": base_dir / "transformer" / "models" / "training",
    }


def _pre_softmax_dense_output_tensor(model):
    """Tensor at the last Dense layer before the final softmax (skips trailing Dropout)."""
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
    return layers[i].output, int(layers[i].units)


def _init_practice_scoring(model_dir: Path) -> None:
    """Build embedding submodel and load learning_prototypes.json when present."""
    global _embedding_model, _prototype_matrix, _prototype_source, _practice_scoring_ready

    _embedding_model = None
    _prototype_matrix = None
    _prototype_source = None
    _practice_scoring_ready = False

    import tensorflow as tf

    try:
        pen_out, emb_dim = _pre_softmax_dense_output_tensor(_model)
        _embedding_model = tf.keras.Model(inputs=_model.input, outputs=pen_out, name="embedding_extractor")
        logger.info(f"Practice embedding model ready (dim={emb_dim})")
    except Exception as e:
        logger.warning(f"Could not build embedding model for practice scoring: {e}")
        _embedding_model = None

    proto_path = model_dir / "learning_prototypes.json"
    if _embedding_model is None:
        logger.warning("Embedding model unavailable; practice scores will use softmax target probability only.")
        return

    emb_dim = int(_embedding_model.output_shape[-1])

    if not proto_path.exists():
        logger.warning(f"No learning_prototypes.json at {proto_path}; practice scores will use softmax fallback.")
        return

    try:
        with open(proto_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("prototypes") or {}
        file_dim = int(data.get("embedding_dim", 0) or 0)
        if file_dim and file_dim != emb_dim:
            logger.warning(
                f"learning_prototypes.json embedding_dim {file_dim} != model embedding {emb_dim}; skipping prototypes."
            )
            return

        mat = np.zeros((len(_class_labels), emb_dim), dtype=np.float32)
        for i, lab in enumerate(_class_labels):
            vec = raw.get(lab) or raw.get(lab.lower()) or raw.get(lab.capitalize())
            if vec is None:
                logger.warning(f"No prototype vector for class '{lab}' in {proto_path}")
                continue
            row = np.asarray(vec, dtype=np.float32).reshape(-1)
            if row.shape[0] != emb_dim:
                logger.warning(f"Prototype for '{lab}' has length {row.shape[0]}, expected {emb_dim}; skipping file.")
                return
            n = float(np.linalg.norm(row))
            if n > 1e-8:
                row = row / n
            mat[i] = row

        _prototype_matrix = mat
        _prototype_source = data.get("prototype_source", "unknown")
        _practice_scoring_ready = True
        logger.info(
            f"Loaded learning prototypes from {proto_path} (source={_prototype_source}, classes={len(_class_labels)})"
        )
    except Exception as e:
        logger.warning(f"Failed to load learning_prototypes.json: {e}")
        _prototype_matrix = None
        _practice_scoring_ready = False


def _resolve_target_class_index(target_label: str) -> int:
    if not _class_labels:
        raise RuntimeError("Class labels not loaded")
    t = (target_label or "").strip().lower()
    for i, lab in enumerate(_class_labels):
        if str(lab).lower() == t:
            return i
    raise ValueError(f"Unknown target_label '{target_label}'. Not in model vocabulary.")


def _normalize_input_layer_batch_shape(config) -> bool:
    """
    Convert Keras 3 H5 InputLayer config keys for older tf.keras loaders.

    Some saved .h5 files store InputLayer shape as `batch_shape`; tf.keras 2.x
    expects `batch_input_shape`. This keeps deployment tolerant if Railway ever
    resolves an older TensorFlow wheel or cached image.
    """
    changed = False

    if isinstance(config, dict):
        if config.get("class_name") == "InputLayer":
            layer_config = config.get("config") or {}
            if "batch_shape" in layer_config and "batch_input_shape" not in layer_config:
                layer_config["batch_input_shape"] = layer_config.pop("batch_shape")
                changed = True

        for value in config.values():
            changed = _normalize_input_layer_batch_shape(value) or changed

    elif isinstance(config, list):
        for item in config:
            changed = _normalize_input_layer_batch_shape(item) or changed

    return changed


def _load_model_with_inputlayer_patch(tf, model_path: Path, custom_objects: Dict):
    """Patch a temporary copy of an H5 model when InputLayer.batch_shape breaks loading."""
    import h5py

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            temp_path = Path(tmp.name)

        shutil.copy2(model_path, temp_path)

        with h5py.File(temp_path, "r+") as h5_file:
            raw_config = h5_file.attrs.get("model_config")
            if raw_config is None:
                raise RuntimeError("H5 model_config attribute is missing")
            if isinstance(raw_config, bytes):
                raw_config = raw_config.decode("utf-8")

            model_config = json.loads(raw_config)
            if not _normalize_input_layer_batch_shape(model_config):
                raise RuntimeError("No InputLayer.batch_shape entry found to patch")

            h5_file.attrs["model_config"] = json.dumps(model_config).encode("utf-8")

        logger.info("Retrying Keras model load with patched InputLayer config")
        return tf.keras.models.load_model(str(temp_path), custom_objects=custom_objects, compile=False)

    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception as cleanup_error:
                logger.warning(f"Could not delete temporary patched model {temp_path}: {cleanup_error}")


def _load_keras_model(tf, model_path: Path, custom_objects: Dict):
    try:
        return tf.keras.models.load_model(str(model_path), custom_objects=custom_objects, compile=False)
    except Exception as exc:
        message = str(exc)
        if "InputLayer" in message and "batch_shape" in message:
            logger.warning(f"Keras H5 InputLayer compatibility issue detected: {message}")
            return _load_model_with_inputlayer_patch(tf, model_path, custom_objects)
        raise


def load_model():
    """
    Load the trained model, normalization parameters, and class labels.
    This function is called once on application startup.
    """
    global _model, _embedding_model, _normalization_params, _class_labels, _model_loaded, _custom_objects, _model_load_error
    global _prototype_matrix, _prototype_source, _practice_scoring_ready

    if _model_loaded:
        logger.info("Model already loaded, skipping...")
        return

    _model_load_error = None

    try:
        # Import TensorFlow only when needed
        import tensorflow as tf

        paths = _get_model_paths()

        # Verify all files exist
        for name, path in paths.items():
            if name == "training_module":
                continue  # This is a directory, not a file
            if not path.exists():
                raise FileNotFoundError(f"{name.capitalize()} file not found at: {path}")

        logger.info("Loading PSL recognition model...")

        # Add the training module to sys.path to import custom layers
        training_module_path = str(paths['training_module'])
        if training_module_path not in sys.path:
            sys.path.insert(0, training_module_path)
        
        # Import custom layers from the transformer training module
        from model_architecture import TCNBlock, TransformerBlock, PositionalEncoding
        
        # Create custom objects dictionary for model loading
        _custom_objects = {
            'TCNBlock': TCNBlock,
            'TransformerBlock': TransformerBlock,
            'PositionalEncoding': PositionalEncoding
        }

        # Load the Keras model with custom objects
        logger.info(f"Loading model from: {paths['model']}")
        _model = _load_keras_model(tf, paths['model'], _custom_objects)
        logger.info(f"Model loaded successfully. Input shape: {_model.input_shape}")

        # Load normalization parameters
        logger.info(f"Loading normalization params from: {paths['normalization']}")
        with open(paths['normalization'], 'r') as f:
            _normalization_params = json.load(f)

        # Convert to numpy arrays for faster computation
        _normalization_params['mean'] = np.array(_normalization_params['mean'])
        _normalization_params['std'] = np.array(_normalization_params['std'])

        # Validate normalization params shape
        if len(_normalization_params['mean']) != 188 or len(_normalization_params['std']) != 188:
            raise ValueError(f"Normalization params must have 188 features, got mean: {len(_normalization_params['mean'])}, std: {len(_normalization_params['std'])}")

        logger.info(f"Normalization params loaded: mean shape {_normalization_params['mean'].shape}, std shape {_normalization_params['std'].shape}")

        # Load metadata (class labels)
        logger.info(f"Loading metadata from: {paths['metadata']}")
        with open(paths['metadata'], 'r') as f:
            metadata = json.load(f)

        # Check for class_names first, then classes (for compatibility)
        _class_labels = metadata.get('class_names', metadata.get('classes', []))

        expected_classes = int(_model.output_shape[-1])
        if len(_class_labels) != expected_classes:
            raise ValueError(f"Class label count mismatch. Model outputs {expected_classes}, metadata has {len(_class_labels)}")

        logger.info(f"Loaded {len(_class_labels)} class labels: {', '.join(_class_labels[:5])}...")

        _model_loaded = True
        _model_load_error = None
        _init_practice_scoring(paths["model"].parent)
        logger.info("PSL model initialization complete!")

    except FileNotFoundError as e:
        logger.warning(f"PSL model files not found: {str(e)}")
        logger.warning("PSL recognition will be disabled. Text-to-PSL animations will still work.")
        logger.info("To enable PSL recognition, train a model using the transformer training pipeline.")
        _model_loaded = False
        _model_load_error = str(e)
        _embedding_model = None
        _prototype_matrix = None
        _practice_scoring_ready = False
    except Exception as e:
        logger.error(f"Failed to load PSL model: {str(e)}")
        logger.warning("PSL recognition will be disabled. Text-to-PSL animations will still work.")
        _model_loaded = False
        _model_load_error = str(e)
        _embedding_model = None
        _prototype_matrix = None
        _practice_scoring_ready = False


def is_model_available() -> bool:
    """Check if the PSL model is loaded and ready for inference."""
    return _model_loaded and _model is not None


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization to the input sequence.

    Args:
        sequence: numpy array of shape (60, 188)

    Returns:
        Normalized sequence of shape (60, 188)
    """
    if _normalization_params is None:
        raise RuntimeError("Normalization parameters not loaded. Call load_model() first.")

    mean = _normalization_params['mean']
    std = _normalization_params['std']

    # Avoid division by zero
    std_safe = np.where(std == 0, 1.0, std)

    # Apply normalization: (x - mean) / std
    normalized = (sequence - mean) / std_safe

    return normalized


def predict_psl(sequence: List[List[float]]) -> Dict:
    """
    Run PSL inference on a sequence of feature vectors.

    Args:
        sequence: List of 60 frames, each containing 188 features.
                 Shape should be (60, 188).

    Returns:
        Dictionary containing:
        - label: str (predicted PSL word)
        - class_id: int (predicted class index)
        - confidence: float (confidence score 0-1)
        - top_predictions: list of top 5 predictions with labels and scores

    Raises:
        ValueError: If input shape is invalid
        RuntimeError: If model is not loaded
    """
    if not _model_loaded:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # Validate input
    if not isinstance(sequence, (list, np.ndarray)):
        raise ValueError("Sequence must be a list or numpy array")

    # Convert to numpy array
    sequence_array = np.array(sequence, dtype=np.float32)

    # Validate shape
    if sequence_array.shape != (60, 188):
        raise ValueError(
            f"Invalid input shape. Expected (60, 188), got {sequence_array.shape}. "
            f"Sequence must contain exactly 60 frames of 188 features each."
        )

    # Check for NaN or Inf values
    if np.isnan(sequence_array).any():
        raise ValueError("Input sequence contains NaN values")
    if np.isinf(sequence_array).any():
        raise ValueError("Input sequence contains infinite values")

    try:
        # Normalize the sequence
        normalized_sequence = normalize_sequence(sequence_array)

        # Add batch dimension: (60, 188) -> (1, 60, 188)
        batch_input = np.expand_dims(normalized_sequence, axis=0)

        # Run inference
        predictions = _model.predict(batch_input, verbose=0)

        # predictions shape: (1, num_classes)
        # Extract the single prediction
        probs = predictions[0]

        # Get predicted class
        predicted_class_id = int(np.argmax(probs))
        predicted_confidence = float(probs[predicted_class_id])
        predicted_label = _class_labels[predicted_class_id]

        # Get top 5 predictions
        top_k = 5
        top_indices = np.argsort(probs)[-top_k:][::-1]  # Descending order

        top_predictions = []
        for idx in top_indices:
            top_predictions.append({
                "label": _class_labels[int(idx)],
                "class_id": int(idx),
                "confidence": float(probs[idx])
            })

        result = {
            "label": predicted_label,
            "class_id": predicted_class_id,
            "confidence": predicted_confidence,
            "top_predictions": top_predictions
        }

        logger.info(f"Prediction: {predicted_label} (confidence: {predicted_confidence:.3f})")

        return result

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise


def score_practice_sequence(
    sequence: List[List[float]],
    target_label: str,
    hands_detected: int = 0,
) -> Dict:
    """
    Score a performed sign against a single target class (embedding cosine vs prototype,
    with softmax target probability as fallback when prototypes are unavailable).
    """
    if not _model_loaded:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    if hands_detected <= 0:
        raise ValueError(
            "No hands detected in the input. Please ensure at least one hand is visible in the camera."
        )

    if not isinstance(sequence, (list, np.ndarray)):
        raise ValueError("Sequence must be a list or numpy array")

    sequence_array = np.array(sequence, dtype=np.float32)
    if sequence_array.shape != (60, 188):
        raise ValueError(
            f"Invalid input shape. Expected (60, 188), got {sequence_array.shape}. "
            f"Sequence must contain exactly 60 frames of 188 features each."
        )
    if np.isnan(sequence_array).any():
        raise ValueError("Input sequence contains NaN values")
    if np.isinf(sequence_array).any():
        raise ValueError("Input sequence contains infinite values")

    target_idx = _resolve_target_class_index(target_label)
    canonical = _class_labels[target_idx]

    normalized_sequence = normalize_sequence(sequence_array)
    batch_input = np.expand_dims(normalized_sequence, axis=0)

    probs = _model.predict(batch_input, verbose=0)[0]
    target_prob = float(probs[target_idx])

    method = "softmax_fallback"
    cosine_val: Optional[float] = None
    score = max(0.0, min(100.0, 100.0 * target_prob))

    if _embedding_model is not None and _prototype_matrix is not None:
        h = _embedding_model.predict(batch_input, verbose=0)[0].astype(np.float64)
        nrm = float(np.linalg.norm(h))
        if nrm > 1e-8:
            h = h / nrm
        pvec = _prototype_matrix[target_idx].astype(np.float64)
        cosine_val = float(np.clip(np.dot(h, pvec), -1.0, 1.0))
        score = max(0.0, min(100.0, 50.0 * (cosine_val + 1.0)))
        method = "embedding"

    return {
        "score": round(float(score), 2),
        "cosine_similarity": None if cosine_val is None else round(cosine_val, 6),
        "target_label": canonical,
        "target_class_id": target_idx,
        "method": method,
        "target_class_probability": round(target_prob, 6),
    }


def get_model_info() -> Dict:
    """
    Get information about the loaded model.

    Returns:
        Dictionary with model metadata
    """
    if not _model_loaded:
        return {"loaded": False, "error": _model_load_error or "Model not loaded"}

    return {
        "loaded": True,
        "input_shape": str(_model.input_shape),
        "output_shape": str(_model.output_shape),
        "num_classes": len(_class_labels),
        "classes": _class_labels,
        "normalization_features": len(_normalization_params['mean']),
        "practice_score_method": "embedding"
        if (_embedding_model is not None and _prototype_matrix is not None)
        else "softmax_fallback",
        "practice_prototype_source": _prototype_source,
    }


# Auto-load model when module is imported (optional, can be called explicitly from main.py)
# Uncomment the line below to auto-load on import:
# load_model()
