from fastapi import FastAPI

from app.database.cosmos import (
    connect_to_cosmos,
    close_cosmos_connection
)

from app.api.routes.chat import router as chat_router

app = FastAPI()


@app.on_event("startup")
async def startup():

    await connect_to_cosmos()


@app.on_event("shutdown")
async def shutdown():

    await close_cosmos_connection()


app.include_router(chat_router)