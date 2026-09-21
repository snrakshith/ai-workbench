from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database import init_db
from app.routes import books


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Library API", version="0.1.0", lifespan=lifespan)

app.include_router(books.router)


@app.get("/", tags=["health"])
async def root():
    return {"message": "Welcome to the Library API", "docs": "/docs"}
