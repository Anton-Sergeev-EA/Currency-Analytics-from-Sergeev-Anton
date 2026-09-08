from pydantic import BaseModel, Field, validator


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="User question about currencies")

    @validator('question')
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('Question cannot be empty')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Какой прогноз по доллару на следующую неделю?"
            }
        }
        