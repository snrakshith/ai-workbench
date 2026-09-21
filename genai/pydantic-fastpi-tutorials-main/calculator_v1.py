"""
FastAPI Calculator - Version 1 (Single Script)

A simple calculator API with all logic in one file.
Demonstrates basic FastAPI usage with Pydantic validation.
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from typing import Literal
from enum import Enum


# Pydantic Models for Request and Response
class OperationType(str, Enum):
    """Enum for supported calculator operations"""
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


class CalculationRequest(BaseModel):
    """Request model for calculation endpoint"""
    operation: OperationType = Field(
        ...,
        description="The mathematical operation to perform"
    )
    num1: float = Field(
        ...,
        description="First operand"
    )
    num2: float = Field(
        ...,
        description="Second operand"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "operation": "add",
                    "num1": 10,
                    "num2": 5
                }
            ]
        }
    }


class CalculationResponse(BaseModel):
    """Response model for calculation endpoint"""
    operation: str = Field(..., description="The operation that was performed")
    num1: float = Field(..., description="First operand")
    num2: float = Field(..., description="Second operand")
    result: float = Field(..., description="The calculated result")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "operation": "add",
                    "num1": 10,
                    "num2": 5,
                    "result": 15
                }
            ]
        }
    }


# Calculator Functions
def add(a: float, b: float) -> float:
    """
    Add two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    return a + b


def subtract(a: float, b: float) -> float:
    """
    Subtract second number from first.

    Args:
        a: First number
        b: Second number

    Returns:
        Difference of a and b
    """
    return a - b


def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Product of a and b
    """
    return a * b


def divide(a: float, b: float) -> float:
    """
    Divide first number by second.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        Quotient of a and b

    Raises:
        ZeroDivisionError: If b is zero
    """
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b


# FastAPI Application
app = FastAPI(
    title="Calculator API - Version 1",
    description="A simple calculator API with basic arithmetic operations (single script version)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.get("/", tags=["General"])
def root():
    """
    Welcome endpoint with API information.
    """
    return {
        "message": "Welcome to Calculator API - Version 1",
        "description": "A simple calculator with basic arithmetic operations",
        "version": "1.0.0",
        "architecture": "Single script (procedural functions)",
        "endpoints": {
            "documentation": "/docs",
            "calculate": "POST /calculate",
            "operations": "GET /operations"
        }
    }


@app.get("/operations", tags=["General"])
def get_operations():
    """
    Get list of all supported operations.
    """
    return {
        "supported_operations": [
            {"operation": "add", "description": "Addition (a + b)", "symbol": "+"},
            {"operation": "subtract", "description": "Subtraction (a - b)", "symbol": "-"},
            {"operation": "multiply", "description": "Multiplication (a × b)", "symbol": "×"},
            {"operation": "divide", "description": "Division (a ÷ b)", "symbol": "÷"}
        ]
    }


@app.post(
    "/calculate",
    response_model=CalculationResponse,
    status_code=status.HTTP_200_OK,
    tags=["Calculator"]
)
def calculate(request: CalculationRequest):
    """
    Perform a calculation based on the operation specified.

    Supports the following operations:
    - **add**: Addition (num1 + num2)
    - **subtract**: Subtraction (num1 - num2)
    - **multiply**: Multiplication (num1 × num2)
    - **divide**: Division (num1 ÷ num2)

    Args:
        request: CalculationRequest with operation and operands

    Returns:
        CalculationResponse with the result

    Raises:
        HTTPException 400: If division by zero is attempted
        HTTPException 422: If invalid operation or parameters
    """
    try:
        # Route to appropriate function based on operation
        if request.operation == OperationType.ADD:
            result = add(request.num1, request.num2)
        elif request.operation == OperationType.SUBTRACT:
            result = subtract(request.num1, request.num2)
        elif request.operation == OperationType.MULTIPLY:
            result = multiply(request.num1, request.num2)
        elif request.operation == OperationType.DIVIDE:
            result = divide(request.num1, request.num2)
        else:
            # This should never happen due to Pydantic validation
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported operation: {request.operation}"
            )

        # Return response
        return CalculationResponse(
            operation=request.operation.value,
            num1=request.num1,
            num2=request.num2,
            result=result
        )

    except ZeroDivisionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {str(e)}"
        )


# Run with: uvicorn calculator_v1:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
