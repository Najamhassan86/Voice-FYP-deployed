"""
PSL Recognition API Router

Provides endpoints for Pakistan Sign Language recognition using the trained model.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, validator, ConfigDict
from typing import List
import logging

from app.services.psl_inference import predict_psl, get_model_info, is_model_available

logger = logging.getLogger(__name__)

router = APIRouter(tags=["PSL Recognition"])


# ==================== Pydantic Models ====================

class PredictionItem(BaseModel):
    """A single prediction with label, class ID, and confidence"""
    label: str = Field(..., description="PSL word label")
    class_id: int = Field(..., ge=0, description="Class index")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")


class PSLRequest(BaseModel):
    """Request body for PSL recognition"""
    sequence: List[List[float]] = Field(
        ...,
        description="Sequence of 60 frames, each with 188 features",
        min_items=60,
        max_items=60
    )
    hands_detected: int = Field(
        default=0,
        description="Number of hands detected in the frames (0, 1, or 2)",
        ge=0,
        le=2
    )

    @validator('sequence')
    def validate_sequence_shape(cls, v):
        """Validate that each frame has exactly 188 features"""
        if len(v) != 60:
            raise ValueError(f"Sequence must contain exactly 60 frames, got {len(v)}")

        for i, frame in enumerate(v):
            if len(frame) != 188:
                raise ValueError(
                    f"Frame {i} must contain exactly 188 features, got {len(frame)}"
                )

        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sequence": [[0.0] * 188 for _ in range(60)],  # Example placeholder
                "hands_detected": 1
            }
        }
    )


class PSLResponse(BaseModel):
    """Response from PSL recognition"""
    label: str = Field(..., description="Predicted PSL word")
    class_id: int = Field(..., description="Predicted class index")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    top_predictions: List[PredictionItem] = Field(
        ...,
        description="Top 5 predictions sorted by confidence"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "label": "careful",
                "class_id": 0,
                "confidence": 0.93,
                "top_predictions": [
                    {"label": "careful", "class_id": 0, "confidence": 0.93},
                    {"label": "dumb", "class_id": 5, "confidence": 0.04},
                    {"label": "far", "class_id": 7, "confidence": 0.02}
                ]
            }
        }
    )


class ModelInfoResponse(BaseModel):
    """Model metadata response"""
    loaded: bool
    input_shape: str = None
    output_shape: str = None
    num_classes: int = None
    classes: List[str] = None
    normalization_features: int = None
    error: str = None


# ==================== API Endpoints ====================

@router.post(
    "/recognize",
    response_model=PSLResponse,
    summary="Recognize PSL sign from landmark sequence",
    description="""
    Recognizes Pakistan Sign Language (PSL) signs from a sequence of hand landmark features.

    **Input Requirements:**
    - 60 frames (2 seconds at 30 FPS)
    - 188 features per frame:
      - Wrist-relative 3D coordinates (63 features per hand × 2)
      - Geometric features (distances, angles, palm ratio)
      - Hand label (left/right one-hot encoding)

    **Output:**
    - Predicted PSL word label
    - Confidence score (0-1)
    - Top 5 predictions with confidence scores

    **Example Usage:**
    ```javascript
    const response = await fetch('/api/psl/recognize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sequence: landmarkSequence })
    });
    const result = await response.json();
    console.log(`Predicted: ${result.label} (${result.confidence})`);
    ```
    """,
    responses={
        200: {
            "description": "Successful recognition",
            "content": {
                "application/json": {
                    "example": {
                        "label": "careful",
                        "class_id": 0,
                        "confidence": 0.93,
                        "top_predictions": [
                            {"label": "careful", "class_id": 0, "confidence": 0.93},
                            {"label": "dumb", "class_id": 5, "confidence": 0.04}
                        ]
                    }
                }
            }
        },
        400: {
            "description": "Invalid input shape or data",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Invalid input shape. Expected (60, 188), got (30, 188)"
                    }
                }
            }
        },
        500: {
            "description": "Server error during inference",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Model inference failed: [error message]"
                    }
                }
            }
        }
    }
)
async def recognize_psl(request: PSLRequest) -> PSLResponse:
    """
    Recognize PSL sign from a sequence of landmark features.

    Args:
        request: PSLRequest containing a (60, 188) sequence and hands_detected count

    Returns:
        PSLResponse with predicted label, confidence, and top predictions

    Raises:
        HTTPException: If input is invalid or inference fails
    """
    try:
        # Check if model is available
        if not is_model_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PSL recognition model is not available. Please check if the model file exists."
            )
        
        # CRITICAL FIX: Check if hands were actually detected
        # This prevents false positives when no hands are in view
        if request.hands_detected == 0:
            logger.warning(f"Ignoring recognition request: No hands detected")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No hands detected in the input. Please ensure at least one hand is visible in the camera."
            )
        
        logger.info(f"Received PSL recognition request with {len(request.sequence)} frames and {request.hands_detected} hand(s)")

        # Run inference
        result = predict_psl(request.sequence)

        logger.info(
            f"PSL recognition successful: {result['label']} "
            f"(confidence: {result['confidence']:.3f})"
        )

        return PSLResponse(**result)

    except ValueError as e:
        # Input validation errors
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    except RuntimeError as e:
        # Model not loaded or runtime errors
        logger.error(f"Runtime error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {str(e)}"
        )

    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected error during PSL recognition: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Get PSL model information",
    description="Returns metadata about the loaded PSL recognition model"
)
async def get_psl_model_info() -> ModelInfoResponse:
    """
    Get information about the loaded PSL model.

    Returns:
        ModelInfoResponse with model metadata
    """
    try:
        info = get_model_info()
        return ModelInfoResponse(**info)

    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return ModelInfoResponse(
            loaded=False,
            error=str(e)
        )


@router.get(
    "/health",
    summary="PSL service health check",
    description="Check if the PSL recognition service is operational"
)
async def psl_health_check():
    """
    Check if PSL service is healthy.

    Returns:
        Status message
    """
    info = get_model_info()
    if info.get("loaded"):
        return {
            "status": "healthy",
            "service": "PSL Recognition",
            "model_loaded": True,
            "num_classes": info.get("num_classes")
        }
    else:
        return {
            "status": "unhealthy",
            "service": "PSL Recognition",
            "model_loaded": False,
            "error": info.get("error", "Model not loaded")
        }
