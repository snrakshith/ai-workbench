from fastapi import APIRouter, HTTPException, Response, status

from app import repository
from app.schemas import Book, BookCreate, BookUpdate

router = APIRouter(prefix="/books", tags=["books"])


@router.get("/", response_model=list[Book])
async def read_books():
    return repository.list_books()


@router.post("/", response_model=Book, status_code=status.HTTP_201_CREATED)
async def add_book(book: BookCreate):
    return repository.create_book(book)


@router.get("/{book_id}", response_model=Book)
async def read_book(book_id: int):
    book = repository.get_book(book_id)
    if book is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found.")
    return book


@router.put("/{book_id}", response_model=Book)
async def replace_book(book_id: int, book: BookCreate):
    updated_book = repository.update_book(book_id, BookUpdate(**book.model_dump()))
    if updated_book is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found.")
    return updated_book


@router.patch("/{book_id}", response_model=Book)
async def edit_book(book_id: int, book: BookUpdate):
    updated_book = repository.update_book(book_id, book)
    if updated_book is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found.")
    return updated_book


@router.delete("/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_book(book_id: int):
    deleted = repository.delete_book(book_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)