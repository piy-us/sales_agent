# import uuid

# from datetime import datetime

# from app.database.cosmos import cosmos_db


# class ConversationRepository:

#     async def save_message(
#         self,
#         conversation_id: str,
#         user_id: str,
#         role: str,
#         content: str,
#     ):

#         item = {
#             "id": str(uuid.uuid4()),
#             "conversationId": conversation_id,
#             "userId": user_id,
#             "role": role,
#             "content": content,
#             "createdAt": (
#                 datetime.utcnow().isoformat()
#             ),
#         }

#         await cosmos_db.container.create_item(
#             item
#         )

#     async def get_messages(
#         self,
#         conversation_id: str,
#         limit: int = 20
#     ):

#         query = """
#         SELECT TOP @limit *
#         FROM c
#         WHERE c.conversationId = @conversationId
#         ORDER BY c.createdAt ASC
#         """

#         parameters = [
#             {
#                 "name": "@conversationId",
#                 "value": conversation_id,
#             },
#             {
#                 "name": "@limit",
#                 "value": limit,
#             },
#         ]

#         items = (
#             cosmos_db.container.query_items(
#                 query=query,
#                 parameters=parameters,
#                 partition_key=conversation_id,
#             )
#         )

#         results = []

#         async for item in items:
#             results.append(item)

#         return results

import uuid

from datetime import datetime

from app.database.cosmos import cosmos_db


class ConversationRepository:

    async def save_message(
        self,
        conversation_id: str,
        user_id: str,
        role: str,
        content: str,
    ):

        item = {
            "id": str(uuid.uuid4()),
            "conversationId": conversation_id,
            "userId": user_id,
            "role": role,
            "content": content,
            "createdAt": (
                datetime.utcnow().isoformat()
            ),
        }

        await cosmos_db.container.create_item(
            item
        )

    async def get_messages(
        self,
        conversation_id: str,
        limit: int = 20
    ):

        query = """
        SELECT TOP @limit *
        FROM c
        WHERE c.conversationId = @conversationId
        ORDER BY c.createdAt ASC
        """

        parameters = [
            {
                "name": "@conversationId",
                "value": conversation_id,
            },
            {
                "name": "@limit",
                "value": limit,
            },
        ]

        items = (
            cosmos_db.container.query_items(
                query=query,
                parameters=parameters,
                partition_key=conversation_id,
            )
        )

        results = []

        async for item in items:
            results.append(item)

        return results

    async def clear_conversation(
        self,
        conversation_id: str,
    ):

        messages = await self.get_messages(
            conversation_id=conversation_id,
            limit=1000
        )

        for msg in messages:

            await cosmos_db.container.delete_item(
                item=msg["id"],
                partition_key=conversation_id,
            )