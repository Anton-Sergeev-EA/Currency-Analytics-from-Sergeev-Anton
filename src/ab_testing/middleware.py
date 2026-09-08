from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import uuid
import logging

from .ab_service import get_ab_service

logger = logging.getLogger(__name__)

class ABTestMiddleware(BaseHTTPMiddleware):
    """
    Middleware для автоматического назначения пользователей на группы A/B-теста.
    """
    
    async def dispatch(self, request: Request, call_next):
        session_id = request.headers.get('X-Session-ID')
        if not session_id:
            session_id = str(uuid.uuid4())
            request.state.session_id = session_id
        
        response = await call_next(request)
        
        response.headers['X-Session-ID'] = session_id
        
        return response
    