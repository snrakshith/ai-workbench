"""
Calculator Module - OOP Implementation

Contains the Calculator class with all arithmetic operations.
This module is independent of FastAPI and can be used in any Python application.
"""

from typing import Dict, Callable


class Calculator:
    """
    A calculator class that performs basic arithmetic operations.

    This class encapsulates all calculator logic using OOP principles,
    making it easy to test, extend, and maintain independently from the API layer.

    Supported Operations:
        - Addition
        - Subtraction
        - Multiplication
        - Division

    Example:
        >>> calc = Calculator()
        >>> result = calc.add(5, 3)
        >>> print(result)
        8.0
        >>> result = calc.calculate("multiply", 4, 7)
        >>> print(result)
        28.0
    """

    def __init__(self):
        """
        Initialize the Calculator.

        Sets up the operation mapping for the calculate method.
        """
        # Map operation names to methods
        self._operations: Dict[str, Callable[[float, float], float]] = {
            "add": self.add,
            "subtract": self.subtract,
            "multiply": self.multiply,
            "divide": self.divide
        }

    def add(self, a: float, b: float) -> float:
        """
        Add two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            The sum of a and b

        Example:
            >>> calc = Calculator()
            >>> calc.add(10, 5)
            15.0
        """
        return a + b

    def subtract(self, a: float, b: float) -> float:
        """
        Subtract second number from first.

        Args:
            a: First number (minuend)
            b: Second number (subtrahend)

        Returns:
            The difference of a and b

        Example:
            >>> calc = Calculator()
            >>> calc.subtract(10, 5)
            5.0
        """
        return a - b

    def multiply(self, a: float, b: float) -> float:
        """
        Multiply two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            The product of a and b

        Example:
            >>> calc = Calculator()
            >>> calc.multiply(10, 5)
            50.0
        """
        return a * b

    def divide(self, a: float, b: float) -> float:
        """
        Divide first number by second.

        Args:
            a: Numerator
            b: Denominator

        Returns:
            The quotient of a and b

        Raises:
            ZeroDivisionError: If b is zero

        Example:
            >>> calc = Calculator()
            >>> calc.divide(10, 5)
            2.0
            >>> calc.divide(10, 0)
            Traceback (most recent call last):
                ...
            ZeroDivisionError: Cannot divide by zero
        """
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return a / b

    def calculate(self, operation: str, num1: float, num2: float) -> float:
        """
        Perform a calculation based on the operation string.

        This is the main method that routes to the appropriate arithmetic operation.

        Args:
            operation: The operation to perform ("add", "subtract", "multiply", "divide")
            num1: First operand
            num2: Second operand

        Returns:
            The result of the calculation

        Raises:
            ValueError: If operation is not supported
            ZeroDivisionError: If dividing by zero

        Example:
            >>> calc = Calculator()
            >>> calc.calculate("add", 10, 5)
            15.0
            >>> calc.calculate("multiply", 3, 4)
            12.0
            >>> calc.calculate("power", 2, 3)
            Traceback (most recent call last):
                ...
            ValueError: Unsupported operation: power
        """
        # Normalize operation name to lowercase
        operation = operation.lower()

        # Check if operation is supported
        if operation not in self._operations:
            raise ValueError(
                f"Unsupported operation: {operation}. "
                f"Supported operations: {', '.join(self._operations.keys())}"
            )

        # Get the appropriate method and call it
        operation_method = self._operations[operation]
        return operation_method(num1, num2)

    def get_supported_operations(self) -> list[str]:
        """
        Get list of all supported operations.

        Returns:
            List of operation names

        Example:
            >>> calc = Calculator()
            >>> calc.get_supported_operations()
            ['add', 'subtract', 'multiply', 'divide']
        """
        return list(self._operations.keys())


# Example usage
if __name__ == "__main__":
    # Create calculator instance
    calc = Calculator()

    # Test all operations
    print("Calculator Testing")
    print("=" * 50)

    # Addition
    result = calc.add(10, 5)
    print(f"10 + 5 = {result}")

    # Subtraction
    result = calc.subtract(10, 5)
    print(f"10 - 5 = {result}")

    # Multiplication
    result = calc.multiply(10, 5)
    print(f"10 × 5 = {result}")

    # Division
    result = calc.divide(10, 5)
    print(f"10 ÷ 5 = {result}")

    # Using calculate method
    print("\nUsing calculate() method:")
    result = calc.calculate("add", 15, 25)
    print(f"15 + 25 = {result}")

    # Test error handling
    print("\nTesting error handling:")
    try:
        result = calc.divide(10, 0)
    except ZeroDivisionError as e:
        print(f"Division by zero: {e}")

    try:
        result = calc.calculate("power", 2, 3)
    except ValueError as e:
        print(f"Invalid operation: {e}")

    # Show supported operations
    print(f"\nSupported operations: {calc.get_supported_operations()}")
