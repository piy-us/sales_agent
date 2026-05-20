# from pydantic import BaseModel
# from datetime import datetime
# from typing import Optional


# class ChatRequest(BaseModel):

#     user_id: str
#     conversation_id: str
#     message: str


# class MessageModel(BaseModel):

#     id: str

#     conversationId: str

#     userId: str

#     role: str

#     content: str

#     createdAt: datetime

#     metadata: Optional[dict] = None

from pydantic import BaseModel


class ChatRequest(BaseModel):

    contact_id: str

    contact_name: str

    message: str


class ChatResponse(BaseModel):

    reply: str


class ClearRequest(BaseModel):

    contact_id: str