# # from fastapi import APIRouter

# # from app.models.schemas import ChatRequest
# # from app.services.chat_service import ChatService

# # router = APIRouter()

# # chat_service = ChatService()


# # @router.post("/chat")
# # async def chat(req: ChatRequest):

# #     await chat_service.save_user_message(
# #         conversation_id=req.conversation_id,
# #         user_id=req.user_id,
# #         message=req.message
# #     )

# #     messages = await chat_service.build_messages(
# #         conversation_id=req.conversation_id,
# #         user_message=req.message
# #     )

# #     # CALL YOUR AGENT HERE
# #     # reply = await agent.ainvoke(messages)

# #     reply = "sample ai response"

# #     await chat_service.save_assistant_message(
# #         conversation_id=req.conversation_id,
# #         user_id=req.user_id,
# #         message=reply
# #     )

# #     return {
# #         "reply": reply
# #     }
# from fastapi import APIRouter

# from app.models.schemas import (
#     ChatRequest,
#     ChatResponse,
#     ClearRequest,
# )

# from app.services.chat_service import (
#     ChatService
# )

# router = APIRouter()

# chat_service = ChatService()


# @router.post(
#     "/chat",
#     response_model=ChatResponse
# )
# async def chat(request: ChatRequest):

#     result = await (
#         chat_service.process_message(

#             conversation_id=(
#                 request.contact_id
#             ),

#             user_id=request.contact_name,

#             user_query=request.message,
#         )
#     )

#     return ChatResponse(
#         reply=result["response"]
#     )


# @router.delete("/history")
# async def clear_history(
#     request: ClearRequest
# ):

#     await (
#         chat_service.clear_history(
#             request.contact_id
#         )
#     )

#     return {
#         "cleared": request.contact_id
#     }


# @router.get("/history/{contact_id}")
# async def get_history(
#     contact_id: str
# ):

#     history = await (
#         chat_service.get_history(
#             contact_id
#         )
#     )

#     return {
#         "contact_id": contact_id,
#         "messages": history,
#     }
from fastapi import APIRouter

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ClearRequest,
)

from app.services.chat_service import (
    ChatService
)

router = APIRouter()

chat_service = ChatService()


@router.post(
    "/chat",
    response_model=ChatResponse
)
async def chat(request: ChatRequest):

    result = await (
        chat_service.process_message(

            conversation_id=(
                request.contact_id
            ),

            user_id=request.contact_name,

            user_query=request.message,
        )
    )

    return ChatResponse(
        reply=result["response"]
    )


@router.get("/history/{contact_id}")
async def get_history(
    contact_id: str
):

    history = await (
        chat_service.get_history(
            contact_id
        )
    )

    return {
        "contact_id": contact_id,
        "messages": history,
    }


@router.delete("/history")
async def clear_history(
    request: ClearRequest
):

    await (
        chat_service.clear_history(
            request.contact_id
        )
    )

    return {
        "cleared": request.contact_id
    }