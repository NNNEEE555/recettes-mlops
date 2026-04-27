from pydantic import BaseModel


class GlobalForecastInput(BaseModel):
    date: str   # format: YYYY-MM


class SegmentForecastInput(BaseModel):
    date: str   # format: YYYY-MM
    segment_type: str
    segment_value: str