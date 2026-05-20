# from app.repositories.conversation_repository import (
#     ConversationRepository
# )

# from app.services.query_router import (
#     QueryRouter
# )

# from app.services.query_rewriter import (
#     QueryRewriter
# )

# from app.services.retrieval_service import (
#     RetrievalService
# )

# from app.services.context_builder import (
#     ContextBuilder
# )

# from app.services.agent_service import (
#     AgentService
# )


# MAX_HISTORY = 20


# class ChatService:

#     def __init__(self):

#         self.repo = ConversationRepository()

#         self.router = QueryRouter()

#         self.rewriter = QueryRewriter()

#         self.retrieval = RetrievalService()

#         self.context_builder = (
#             ContextBuilder()
#         )

#         self.agent_service = (
#             AgentService()
#         )

#     async def process_message(
#         self,
#         conversation_id: str,
#         user_id: str,
#         user_query: str,
#     ):

#         # LOAD HISTORY
#         history = await (
#             self.repo.get_messages(
#                 conversation_id=conversation_id,
#                 limit=MAX_HISTORY
#             )
#         )

#         use_rag = (
#             self.router.should_use_rag(
#                 user_query
#             )
#         )

#         context = ""

#         rewritten_query = user_query

#         # RAG PIPELINE
#         if use_rag:

#             rewritten_query = (
#                 await self.rewriter.rewrite(
#                     history,
#                     user_query
#                 )
#             )

#             docs = await (
#                 self.retrieval.retrieve(
#                     rewritten_query
#                 )
#             )

#             context = (
#                 self.context_builder
#                 .build_context(docs)
#             )

#         # SAVE USER MESSAGE
#         await self.repo.save_message(
#             conversation_id=conversation_id,
#             user_id=user_id,
#             role="user",
#             content=user_query,
#         )

#         # AGENT GENERATION
#         response = await (
#             self.agent_service
#             .generate_response(
#                 user_query=user_query,
#                 conversation_history=history,
#                 rag_context=context,
#             )
#         )

#         # SAVE AI RESPONSE
#         await self.repo.save_message(
#             conversation_id=conversation_id,
#             user_id=user_id,
#             role="assistant",
#             content=response,
#         )
        
#         return {
#             "response": response,
#             "used_rag": use_rag,
#             "rewritten_query": rewritten_query,
#         }
        
from app.repositories.conversation_repository import (
    ConversationRepository
)

from app.services.query_router import (
    QueryRouter
)

from app.services.query_rewriter import (
    QueryRewriter
)

from app.services.retrieval_service import (
    RetrievalService
)

from app.services.context_builder import (
    ContextBuilder
)

from app.services.agent_service import (
    AgentService
)


MAX_HISTORY = 20


class ChatService:

    def __init__(self):

        self.repo = ConversationRepository()

        self.router = QueryRouter()

        self.rewriter = QueryRewriter()

        self.retrieval = RetrievalService()

        self.context_builder = (
            ContextBuilder()
        )

        self.agent_service = (
            AgentService()
        )

    async def process_message(
        self,
        conversation_id: str,
        user_id: str,
        user_query: str,
    ):

        # LOAD HISTORY
        history = await (
            self.repo.get_messages(
                conversation_id=conversation_id,
                limit=MAX_HISTORY
            )
        )

        # DETERMINE IF RAG IS NEEDED
        use_rag = (
            self.router.should_use_rag(
                user_query
            )
        )

        context = ""

        rewritten_query = user_query

        # RAG PIPELINE
        if use_rag:

            rewritten_query = (
                await self.rewriter.rewrite(
                    history,
                    user_query
                )
            )

            docs = await (
                self.retrieval.retrieve(
                    rewritten_query
                )
            )

            context = (
                self.context_builder
                .build_context(docs)
            )

        # SAVE USER MESSAGE
        await self.repo.save_message(
            conversation_id=conversation_id,
            user_id=user_id,
            role="user",
            content=user_query,
        )

        # GENERATE RESPONSE
        response = await (
            self.agent_service
            .generate_response(
                user_query=user_query,
                conversation_history=history,
                rag_context=context,
            )
        )

        # SAVE ASSISTANT RESPONSE
        await self.repo.save_message(
            conversation_id=conversation_id,
            user_id=user_id,
            role="assistant",
            content=response,
        )

        return {
            "response": response,
            "used_rag": use_rag,
            "rewritten_query": rewritten_query,
        }

    async def get_history(
        self,
        conversation_id: str,
    ):

        history = await (
            self.repo.get_messages(
                conversation_id=conversation_id,
                limit=100
            )
        )

        return [
            {
                "role": msg["role"],
                "content": msg["content"],
            }
            for msg in history
        ]

    async def clear_history(
        self,
        conversation_id: str,
    ):

        await self.repo.clear_conversation(
            conversation_id
        )