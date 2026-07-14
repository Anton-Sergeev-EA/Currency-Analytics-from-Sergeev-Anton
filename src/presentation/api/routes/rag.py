from fastapi import APIRouter
from src.application.services.rag_service import RAGService
from src.presentation.api.schemas.request import AskRequest
from src.presentation.api.schemas.response import AskResponse

router = APIRouter(tags=["RAG"])
service = RAGService()


@router.post("/api/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Ask a question to the RAG system."""
    response = service.ask(request.question)

    return AskResponse(
        answer=response["answer"],
        type=response["type"],
        confidence=response.get("confidence")
    )
