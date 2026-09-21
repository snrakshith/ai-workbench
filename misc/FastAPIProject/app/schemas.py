from datetime import datetime

from pydantic import BaseModel, Field


class BookBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    author: str = Field(..., min_length=1, max_length=200)
    isbn: str = Field(..., min_length=1, max_length=32)
    published_year: int | None = Field(default=None, ge=0, le=9999)
    genre: str | None = Field(default=None, max_length=100)
    available: bool = True


class BookCreate(BookBase):
    pass


class BookUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    author: str | None = Field(default=None, min_length=1, max_length=200)
    isbn: str | None = Field(default=None, min_length=1, max_length=32)
    published_year: int | None = Field(default=None, ge=0, le=9999)
    genre: str | None = Field(default=None, max_length=100)
    available: bool | None = None


class Book(BookBase):
    id: int
    created_at: datetime
    updated_at: datetime

