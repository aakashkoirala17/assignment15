"""Standardized error responses."""

from typing import Any, Dict, Optional
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class APIErrorResponse(BaseModel):
    status: str = "error"
    error: ErrorDetail
