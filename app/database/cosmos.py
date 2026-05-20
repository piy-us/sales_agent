from azure.cosmos.aio import CosmosClient

from azure.cosmos import PartitionKey

from app.core.config import settings


class CosmosDB:

    client = None

    database = None

    container = None


cosmos_db = CosmosDB()


async def connect_to_cosmos():

    cosmos_db.client = CosmosClient(
        settings.COSMOS_ENDPOINT,
        credential=settings.COSMOS_KEY
    )

    database = await (
        cosmos_db.client
        .create_database_if_not_exists(
            id=settings.COSMOS_DATABASE
        )
    )

    container = await (
        database.create_container_if_not_exists(
            id=settings.COSMOS_CONTAINER,
            partition_key=PartitionKey(
                path="/conversationId"
            ),
        )
    )

    cosmos_db.database = database

    cosmos_db.container = container


async def close_cosmos_connection():

    await cosmos_db.client.close()