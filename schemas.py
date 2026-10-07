"""Data shapes: what Gemini returns (Extracted*) and what goes in the sheet (Item)."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ItemType = Literal["news", "paper", "release"]
ModelType = Literal["LLM", "vision", "audio", "multimodal", "other", "N/A"]


class ExtractedItem(BaseModel):
    """One item as classified by the LLM. Dates are NOT asked from the LLM."""

    url: str = Field(description="Exact URL of the source search result this item is based on")
    type: ItemType
    model_type: ModelType = Field(description="Kind of AI model involved, or N/A if none")
    summary: str = Field(description="One or two plain sentences")
    is_new_model_release: bool = Field(
        description="True only if the source says a brand-new AI model was officially released/launched"
    )
    model_name: str | None = Field(
        default=None, description="Name of the released model when is_new_model_release is true"
    )


class ExtractionResult(BaseModel):
    items: list[ExtractedItem]


class Item(BaseModel):
    """A validated, dated item ready for the sheet."""

    published_at: datetime
    type: ItemType
    model_type: ModelType
    summary: str
    url: str
    is_new_model_release: bool = False
    model_name: str | None = None

    def to_row(self) -> list[str]:
        return [
            self.published_at.strftime("%Y-%m-%d"),
            self.published_at.strftime("%H:%M"),
            self.type,
            self.model_type,
            self.summary,
            self.url,
        ]
