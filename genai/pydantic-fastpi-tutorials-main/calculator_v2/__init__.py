"""
Calculator V2 Package

A modular calculator implementation using OOP principles.
Separates business logic (Calculator class) from API layer (FastAPI).
"""

from .calculator import Calculator

__all__ = ["Calculator"]
__version__ = "2.0.0"
