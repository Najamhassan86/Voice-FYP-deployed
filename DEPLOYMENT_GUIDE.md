# Deployment Guide

## Architecture

- `voice-frontend` deploys to Vercel
- Repository root deploys to Railway for the FastAPI backend

## Required model file

Place the trained model at:

`transformer/models/saved_models/tcn_transformer_20260417_011511/best_model.h5`

The backend loads `normalization_params.json` from the same folder and metadata from:

`transformer/data/extracted_landmarks/metadata.json`

## Railway

- Root directory: repository root
- Service type: Dockerfile
- Exposed health path: `/health`

Environment variables:

- `CORS_ORIGINS=https://your-vercel-app.vercel.app,http://localhost:5173`
- `MODEL_DIR=transformer/models/saved_models/tcn_transformer_20260417_011511`

## Vercel

- Root directory: `voice-frontend`
- Framework preset: `Vite`

Environment variables:

- `VITE_API_BASE_URL=https://your-railway-backend.up.railway.app`
