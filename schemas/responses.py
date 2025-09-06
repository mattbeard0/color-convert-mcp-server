from pydantic import BaseModel
from typing import Optional, Dict

class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict] = None

class SuccessResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None
