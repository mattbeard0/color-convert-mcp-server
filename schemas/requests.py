from pydantic import BaseModel, Field
from typing import Literal

class ColorConvertRequest(BaseModel):
    code: str = Field(..., description="The CSS color code to convert")
    target: Literal["hex", "rgb", "hsl", "hwb", "lab", "lch", "oklab", "oklch", "named"] = Field(..., description="The target color code format to convert to")


