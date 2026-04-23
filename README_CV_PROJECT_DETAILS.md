# D-VOICE Project Detailed Dossier (For CV/Portfolio Generation)

This document is prepared so an LLM (for example Claude) can generate accurate CV bullets, portfolio descriptions, interview summaries, and technical writeups for this project.

## 1) Project Identity

- Project Name: D-VOICE
- Domain: Assistive AI / Computer Vision / Sign Language Recognition
- Primary Goal: Real-time Pakistan Sign Language (PSL) recognition with educational support through an interactive learning module.
- High-level Outcome: A full-stack system that converts live sign gestures into text and speech, while also enabling guided learning and feedback.

## 2) Problem We Are Solving

People who rely on sign language often face communication barriers in mainstream environments. This project targets that accessibility gap by creating:

1. A real-time PSL-to-text translator using webcam input.
2. A sentence-building interface with optional speech output.
3. A Learn module that helps users practice signs and receive model-driven feedback.

## 3) End-to-End System Overview

### Input Layer

- Webcam video stream in browser.
- Continuous frame capture from user gestures.

### Vision + Feature Layer

- MediaPipe Hands detects up to 2 hands per frame.
- Landmark coordinates are transformed into a fixed feature vector.
- Feature dimension per frame: 188.

### Temporal Modeling Layer

- Features are buffered into sequences.
- Sequence length: 60 frames.
- Sequence tensor shape: (60, 188).

### Model Inference Layer

- Backend serves a trained temporal model (TCN + Transformer).
- Returns predicted class label, confidence, and top-k predictions.

### Application Layer

- React frontend shows live prediction and confidence.
- Supports sentence accumulation and text-to-speech.
- Learn module selects target word and evaluates live attempts.

## 4) Full Technology Stack

## Programming Languages

- Python
- JavaScript / JSX

## ML / CV

- TensorFlow / Keras
- MediaPipe Hands
- NumPy
- scikit-learn (metrics/splitting)

## Backend

- FastAPI
- Uvicorn
- Pydantic schemas

## Frontend

- React
- Vite
- Axios
- Browser APIs (MediaDevices, SpeechSynthesis)

## Tooling / Engineering

- Git / GitHub
- Feature branches + PR workflow
- Python virtual environments
- Metrics artifacts (JSON/CSV/image outputs)

## 5) Modeling Approach and Why

The system uses a TCN + Transformer architecture for temporal sequence classification.

Why this combination:

1. TCN captures local temporal gesture dynamics effectively.
2. Transformer captures broader sequence context and long-range dependencies.
3. Hybrid design improves robustness over static frame-only classification.

### Feature Engineering Summary

Each frame is converted into a 188-D vector that includes:

1. Wrist-relative landmark coordinates.
2. Hand geometry descriptors (distances, angular relations, orientation cues).
3. Handedness encoding.

This reduces noise from raw pixels and makes training/inference faster and more stable.

## 6) Training and Data Pipeline

### Data Flow

1. Raw videos organized by class label.
2. Landmark extraction for each clip.
3. Sequence generation and serialization.
4. Stratified train/validation/test splitting.
5. Augmentation on training split only.
6. Normalization fitted on training and reused for val/test/inference.
7. Best model checkpoint + metadata + normalization parameters saved.

### Anti-Leakage Practice Implemented

- Split before augmentation.
- Augment only training samples.
- Keep validation/test untouched to preserve fair evaluation.

## 7) Current Active Runtime Model (What is being used now)

- Active model family: TCN + Transformer
- Active run artifact folder:
  - transformer/models/saved_models/tcn_transformer_20260417_011511
- Active output classes in current deployed run: 9
  - careful, cheap, crazy, dangerous, decent, dumb, fantastic, far, fearful

## 8) Evaluated Metrics (Computed from Current Active Model)

These metrics were computed from the active run artifacts and test split reconstruction.

### Core Accuracy Metrics

- Test Accuracy: 0.9824561403508771 (98.25%)
- F1 Micro: 0.9824561403508771 (98.25%)
- F1 Macro: 0.9824915824915826 (98.25%)
- F1 Weighted: 0.9822434875066454 (98.22%)
- Test Top-3 Accuracy: 1.0 (100%)
- Test Top-5 Accuracy: 1.0 (100%)
- Test Loss: 0.7634549140930176

### Confidence Metrics (Predicted-Class Confidence on Test Set)

- Mean Confidence: 0.9398595690727234 (93.99%)
- Median Confidence: 0.9448371529579163 (94.48%)
- Min Confidence: 0.8208748698234558 (82.09%)
- Max Confidence: 0.9587360620498657 (95.87%)
- Mean Confidence (Correct Predictions): 0.9411472678184509 (94.11%)
- Mean Confidence (Incorrect Predictions): 0.8677504658699036 (86.78%)

### Per-Class Metrics (Precision / Recall / F1)

- careful: 1.00 / 0.8333 / 0.9091
- cheap: 1.00 / 1.00 / 1.00
- crazy: 1.00 / 1.00 / 1.00
- dangerous: 1.00 / 1.00 / 1.00
- decent: 1.00 / 1.00 / 1.00
- dumb: 0.8750 / 1.00 / 0.9333
- fantastic: 1.00 / 1.00 / 1.00
- far: 1.00 / 1.00 / 1.00
- fearful: 1.00 / 1.00 / 1.00

## 9) Training Summary (Current Active Run)

- Model parameters: 197,321
- Total epochs: 59
- Training time: 168.01 seconds
- Final train accuracy: 0.9721 (97.21%)
- Final validation accuracy: 0.9706 (97.06%)
- Best validation accuracy observed: 1.0 (100%)

## 10) Dataset Statements for CV (Important Clarification)

You requested CV wording: "32 words" and "25+ videos per word".

Use this distinction clearly in CV text:

1. Vocabulary scope target: 32-word PSL system.
2. Collection scale statement: 25+ videos per word (minimum expected size >= 800 clips).
3. Current active evaluated run in this code state: 9 classes, 227 raw extracted samples.

This keeps CV claims truthful while still reflecting broader project scope.

## 11) Learn Module (What We Implemented)

The Learn module is not mock-only anymore; it uses real inference flow.

Implemented behavior:

1. Uses the same PSL recognition pipeline as the live translator.
2. Loads active model classes from backend model-info endpoint.
3. Lets user select target word and practice via webcam.
4. Produces prediction-based feedback and confidence-linked similarity.
5. Tracks progress/session stats in frontend state/local storage.
6. Handles backend/model sync failures with retry and error messaging.
7. Added runtime resilience to prevent "stuck" recognition behavior.

## 12) Key Engineering Challenges Solved

1. Model-version mismatch between old and new artifacts.
2. Class-map mismatch between backend and frontend.
3. Normalization consistency across train and inference.
4. Real-time recognition loop stability issues in Learn mode.
5. Multi-process local port conflicts (stale uvicorn instances).
6. Safer backend discovery/fallback in frontend API integration.

## 13) Deployment/Runtime Architecture

1. Frontend (React + Vite) captures camera + sends sequence inference requests.
2. Backend (FastAPI) loads best model checkpoint and normalization metadata.
3. Inference endpoint responds with label and confidence distribution.
4. Frontend displays current sign, sentence output, and learning feedback.

## 14) CV-Ready Bullet Drafts

- Built an end-to-end real-time Pakistan Sign Language recognition platform using MediaPipe landmark extraction and a TCN+Transformer sequence model.
- Engineered a temporal inference pipeline (60x188 sequence input) with robust train/val/test splitting, augmentation policy, and normalization parity across deployment.
- Achieved 98.25% test accuracy and ~98.25% macro-F1 on the active deployed model, with ~94% mean prediction confidence on test predictions.
- Developed a full-stack product (FastAPI + React/Vite) with live sign-to-text, speech output, and an interactive learning module powered by real model inference.

## 15) Interview Summary (Short)

I built a production-style PSL recognition system where webcam gestures are converted to text/speech in real time. I used MediaPipe-based landmark features and trained a TCN+Transformer model for temporal sign understanding. I then deployed the model through FastAPI and integrated it into a React app with both translation and learning workflows. The active run achieved 98.25% test accuracy and ~98.25% macro-F1, and I resolved practical deployment issues like class mismatches, normalization drift risk, and stale process/port conflicts.

## 16) Metrics Artifact Sources

Primary files used for computed and logged metrics:

- transformer/models/saved_models/tcn_transformer_20260417_011511/test_results.json
- transformer/models/saved_models/tcn_transformer_20260417_011511/training_summary.json
- transformer/models/saved_models/tcn_transformer_20260417_011511/cv_metrics_detailed.json
- transformer/data/extracted_landmarks/metadata.json

---

If this text is being passed to Claude for CV generation, ask it to produce:

1. ATS-friendly 3-4 bullet CV version.
2. Detailed portfolio paragraph version.
3. STAR interview answer with quantified metrics.
4. A "scope vs active-run" note so claims remain technically accurate.
