import sqlite3
from typing import Any

from fastapi import HTTPException, status

from app.database import get_connection
from app.schemas import BookCreate, BookUpdate


def _book_from_row(row: sqlite3.Row) -> dict[str, Any]:
    book = dict(row)
    book["available"] = bool(book["available"])
    return book


def list_books() -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute("SELECT * FROM books ORDER BY id").fetchall()
    return [_book_from_row(row) for row in rows]


def get_book(book_id: int) -> dict[str, Any] | None:
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM books WHERE id = ?", (book_id,)).fetchone()
    return _book_from_row(row) if row else None


def create_book(book: BookCreate) -> dict[str, Any]:
    try:
        with get_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO books (title, author, isbn, published_year, genre, available)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    book.title,
                    book.author,
                    book.isbn,
                    book.published_year,
                    book.genre,
                    int(book.available),
                ),
            )
            connection.commit()
            created_id = cursor.lastrowid
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A book with this ISBN already exists.",
        ) from exc

    created_book = get_book(created_id)
    if created_book is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Book was not created.")
    return created_book


def update_book(book_id: int, book: BookUpdate) -> dict[str, Any] | None:
    updates = book.model_dump(exclude_unset=True)
    if not updates:
        return get_book(book_id)

    if "available" in updates and updates["available"] is not None:
        updates["available"] = int(updates["available"])

    set_clause = ", ".join(f"{field} = ?" for field in updates)
    values = list(updates.values())

    try:
        with get_connection() as connection:
            cursor = connection.execute(
                f"UPDATE books SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (*values, book_id),
            )
            connection.commit()
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A book with this ISBN already exists.",
        ) from exc

    if cursor.rowcount == 0:
        return None
    return get_book(book_id)


def delete_book(book_id: int) -> bool:
    with get_connection() as connection:
        cursor = connection.execute("DELETE FROM books WHERE id = ?", (book_id,))
        connection.commit()
    return cursor.rowcount > 0