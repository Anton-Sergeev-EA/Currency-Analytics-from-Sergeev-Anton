from fastapi import APIRouter
from src.application.services.rag_service import RAGService
from src.presentation.api.schemas.request import AskRequest
from src.presentation.api.schemas.response import AskResponse

router = APIRouter(tags=["RAG"])
service = RAGService()


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    response = await service.ask(request.question)

    if isinstance(response, str):
        return AskResponse(
            answer=response,
            type="general",
            confidence=1.0
        )

    return AskResponse(
        answer=response.get("answer", str(response)),
        type=response.get("type", "general"),
        confidence=response.get("confidence", 1.0)
    )
