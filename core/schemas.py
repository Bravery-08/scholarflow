from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field


class DataType(str, Enum):
    chart = "chart"
    diagram = "diagram"
    table = "table"
    photo = "photo"
    equation = "equation"
    other = "other"


class FigureCaption(BaseModel):
    description: str = Field(..., description="What this figure shows")
    data_type: DataType = Field(..., description="Type of visual content")
    key_findings: list[str] = Field(..., description="Up to 3 key findings")
    searchable_keywords: list[str] = Field(..., description="Keywords for retrieval")


class SynthesisResponse(BaseModel):
    answer: str
    citations: list[int]
    confidence: float


class EvalResult(BaseModel):
    faithfulness: float
    answer_relevance: float
    context_precision: float
    flags: list[str] = []