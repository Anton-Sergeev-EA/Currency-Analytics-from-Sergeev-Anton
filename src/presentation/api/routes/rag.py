from fastapi import APIRouter
from src.application.services.rag_service import RAGService
from src.presentation.api.schemas.request import AskRequest
from src.presentation.api.schemas.response import AskResponse

router = APIRouter(tags=["RAG"])
service = RAGService()


@router.post("/api/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Ask a question to the RAG system."""
    response = await service.ask(request.question)

    # Если сервис вернул просто строку, заворачиваем её в правильный формат
    if isinstance(response, str):
        return AskResponse(
            answer=response,
            type="general",
            confidence=1.0
        )

    # Если это всё-таки словарь
    return AskResponse(
        answer=response.get("answer", str(response)),
        type=response.get("type", "general"),
        confidence=response.get("confidence", 1.0)
    )

