# Testing Guide - BSP AI Assistant

## Quick Start Testing Guide

This guide helps developers understand and work with the BSP AI Assistant test suite.

## Table of Contents
1. [Setup and Installation](#setup-and-installation)
2. [Running Tests](#running-tests)
3. [Writing New Tests](#writing-new-tests)
4. [Test Structure](#test-structure)
5. [Mocking Guidelines](#mocking-guidelines)
6. [Debugging Tests](#debugging-tests)
7. [Best Practices](#best-practices)

## Setup and Installation

### Prerequisites
- Python 3.12+
- Virtual environment activated
- All project dependencies installed

### Installation Steps
```bash
# Navigate to project directory
cd c:\Users\robertrita\Workspace\CODES\_MVP\chainlitgpt

# Activate virtual environment
.\venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Install dependencies (if not already done)
pip install -r requirements.txt

# Verify test environment
python -m pytest --version
```

## Running Tests

### Basic Test Commands

```bash
# Run all tests
python run_tests.py unit

# Run with coverage report
python run_tests.py coverage

# Run specific test file
python run_tests.py tests/test_utils.py

# Run in verbose mode (detailed output)
python run_tests.py verbose

# Run specific test class
python -m pytest tests/test_app.py::TestHeaderAuthCallback -v

# Run specific test method
python -m pytest tests/test_utils.py::TestTruncate::test_truncate_no_truncation_needed -v
```

### Advanced Test Options

```bash
# Run tests with specific markers
python -m pytest -m "not slow" -v

# Run tests and stop on first failure
python -m pytest -x tests/

# Run tests with maximum verbosity
python -m pytest -vv tests/

# Run tests and show local variables on failure
python -m pytest -l tests/

# Generate HTML coverage report
python -m pytest --cov=app --cov=utils --cov-report=html tests/
```

## Writing New Tests

### Test File Structure

When adding new functionality, create corresponding test files following this pattern:

```python
# tests/test_new_module.py
"""
Unit tests for new_module.py - Description of module purpose
Tests specific functionality like X, Y, and Z
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.new_module import function_to_test

class TestNewFunction:
    """Test cases for new_function."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        self.test_data = {"key": "value"}
    
    def test_new_function_success(self):
        """Test successful execution of new_function."""
        # Arrange
        input_data = "test input"
        expected_output = "expected result"
        
        # Act
        result = function_to_test(input_data)
        
        # Assert
        assert result == expected_output
    
    def test_new_function_error_handling(self):
        """Test error handling in new_function."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
```

### Test Method Naming Convention

Follow this naming pattern for test methods:
- `test_function_name_success()` - Happy path testing
- `test_function_name_error_handling()` - Error scenarios
- `test_function_name_edge_case()` - Boundary conditions
- `test_function_name_with_specific_condition()` - Specific scenarios

### Test Documentation

Each test should have:
1. **Purpose comment** - What the test validates
2. **Test data description** - What inputs are used
3. **Expected result** - What should happen
4. **Assertions** - What specific checks are made

Example:
```python
def test_truncate_at_word_boundary(self):
    """
    Purpose: Test intelligent word boundary truncation
    Test Data: Text requiring truncation at word boundaries
    Expected Result: Clean truncation without partial words
    Assertions: Word boundary respect, proper length
    """
    text = "This is a long sentence that needs truncation"
    limit = 20
    
    result = truncate(text, limit)
    
    assert len(result) <= limit + 3  # +3 for ellipsis
    assert result.endswith("...")
    assert not result[:-3].endswith(" ")  # No trailing space before ellipsis
```

## Test Structure

### Arrange-Act-Assert Pattern

Structure all tests using the AAA pattern:

```python
def test_example_function(self):
    """Test description."""
    # Arrange - Set up test data and mocks
    input_data = "test input"
    expected_output = "expected result"
    mock_dependency = Mock()
    
    # Act - Execute the function under test
    with patch('module.dependency', mock_dependency):
        result = function_under_test(input_data)
    
    # Assert - Verify the results
    assert result == expected_output
    mock_dependency.assert_called_once_with(input_data)
```

### Test Class Organization

Group related tests into classes:

```python
class TestUtilityFunction:
    """Test cases for utility_function with different scenarios."""
    
    def setup_method(self):
        """Common setup for all test methods in this class."""
        self.common_data = {}
    
    def teardown_method(self):
        """Cleanup after each test method."""
        # Clean up resources if needed
        pass
    
    def test_scenario_one(self):
        """Test first scenario."""
        pass
    
    def test_scenario_two(self):
        """Test second scenario."""
        pass
```

## Mocking Guidelines

### When to Use Mocks

Mock external dependencies such as:
- **API calls** (Azure AI, LiteLLM)
- **File operations** (reading/writing files)
- **Database operations**
- **Network requests**
- **Chainlit framework calls**
- **Environment variables**

### Common Mock Patterns

#### 1. Function Mocking
```python
@patch('module.external_function')
def test_with_function_mock(self, mock_external):
    mock_external.return_value = "mocked result"
    result = function_under_test()
    assert result == "expected based on mock"
```

#### 2. Class Mocking
```python
@patch('module.ExternalClass')
def test_with_class_mock(self, mock_class):
    mock_instance = mock_class.return_value
    mock_instance.method.return_value = "mocked result"
    
    result = function_under_test()
    mock_class.assert_called_once()
```

#### 3. Async Function Mocking
```python
@patch('module.async_function')
async def test_async_function(self, mock_async):
    mock_async.return_value = AsyncMock()
    mock_async.return_value.return_value = "async result"
    
    result = await async_function_under_test()
    assert result == "expected result"
```

#### 4. Context Manager Mocking
```python
@patch('builtins.open', new_callable=mock_open, read_data='file content')
def test_file_reading(self, mock_file):
    result = function_that_reads_file()
    mock_file.assert_called_once_with('expected_file.txt', 'r')
```

#### 5. Environment Variable Mocking
```python
@patch.dict('os.environ', {'ENV_VAR': 'test_value'})
def test_with_env_var(self):
    result = function_using_env_var()
    assert result == "expected based on env var"
```

### Chainlit-Specific Mocking

For Chainlit applications, mock these common components:

```python
# Session mocking
@patch('chainlit.user_session.get')
@patch('chainlit.user_session.set')
def test_chainlit_session(self, mock_set, mock_get):
    mock_get.return_value = {"key": "value"}
    # Test code here
    mock_set.assert_called_with("key", "new_value")

# Message mocking
@patch('chainlit.Message')
async def test_message_sending(self, mock_message_class):
    mock_message = Mock()
    mock_message.send = AsyncMock()
    mock_message_class.return_value = mock_message
    
    await function_that_sends_message()
    mock_message.send.assert_called_once()
```

## Debugging Tests

### Common Test Failures and Solutions

#### 1. Mock Not Called
```python
# Problem: mock.assert_called_once() fails
# Solution: Check if function is actually calling the mocked dependency

# Debug with:
print(f"Mock called: {mock_function.called}")
print(f"Call count: {mock_function.call_count}")
print(f"Call args: {mock_function.call_args_list}")
```

#### 2. Async/Await Issues
```python
# Problem: RuntimeError: coroutine was never awaited
# Solution: Ensure async functions are properly awaited in tests

# Correct pattern:
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function_under_test()
    assert result == expected
```

#### 3. Session State Issues
```python
# Problem: Tests interfere with each other
# Solution: Proper cleanup in setup/teardown

def setup_method(self):
    """Reset state before each test."""
    # Clear any global state
    
def teardown_method(self):
    """Clean up after each test."""
    # Reset mocks, clear caches
```

### Debugging Commands

```bash
# Run single test with maximum verbosity
python -m pytest tests/test_file.py::TestClass::test_method -vv -s

# Run test with debugger (pdb)
python -m pytest tests/test_file.py::TestClass::test_method --pdb

# Show print statements in tests
python -m pytest tests/test_file.py -s

# Show local variables on failure
python -m pytest tests/test_file.py -l

# Run with coverage and show missing lines
python -m pytest tests/test_file.py --cov=module --cov-report=term-missing
```

## Best Practices

### Test Organization

1. **One assertion per test** (when possible)
2. **Clear test names** that describe the scenario
3. **Group related tests** in classes
4. **Use setup/teardown** for common test data
5. **Mock external dependencies** consistently

### Test Data Management

```python
# Good: Use fixtures for reusable test data
@pytest.fixture
def sample_config():
    return {
        "model_deployment": "test/model",
        "model_provider": "test_provider"
    }

def test_with_fixture(self, sample_config):
    result = process_config(sample_config)
    assert result is not None
```

### Error Testing

Always test error conditions:

```python
def test_function_with_invalid_input(self):
    """Test function handles invalid input gracefully."""
    with pytest.raises(ValueError, match="Invalid input"):
        function_under_test("invalid_input")

def test_function_with_network_error(self):
    """Test function handles network errors."""
    with patch('requests.get', side_effect=ConnectionError()):
        result = function_making_network_call()
        assert result is None  # or appropriate error handling
```

### Performance Testing

For performance-critical functions:

```python
import time

def test_function_performance(self):
    """Test function completes within time limit."""
    start_time = time.time()
    
    result = expensive_function()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    assert execution_time < 1.0  # Should complete within 1 second
    assert result is not None
```

### Coverage Guidelines

- **Aim for 95%+ coverage** on new code
- **Test all code paths** including error branches
- **Focus on critical business logic** first
- **Don't chase 100% coverage** at the expense of test quality

### Maintenance

1. **Update tests** when changing functionality
2. **Remove obsolete tests** when features are removed
3. **Refactor tests** to keep them maintainable
4. **Review test failures** promptly in CI/CD
5. **Keep test dependencies** up to date

## Test Environment Configuration

### pytest.ini Configuration

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers
asyncio_mode = auto
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

### conftest.py for Shared Fixtures

```python
# conftest.py
import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_chainlit_session():
    """Provide a mocked Chainlit session for tests."""
    session_mock = Mock()
    session_mock.get.return_value = {}
    session_mock.set.return_value = None
    return session_mock

@pytest.fixture
def sample_llm_config():
    """Provide sample LLM configuration for tests."""
    return [
        {
            "model_deployment": "gpt-4",
            "model_provider": "azure",
            "api_endpoint": "https://test.openai.azure.com/"
        }
    ]
```

## Integration with Development Workflow

### Pre-commit Testing

```bash
# Run before committing code
python run_tests.py unit

# If tests pass, commit
git add .
git commit -m "Feature: Add new functionality with tests"
```

### CI/CD Integration

Ensure your GitHub Actions or other CI/CD includes:

```yaml
name: Run Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.12
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: python run_tests.py unit
    - name: Generate coverage
      run: python run_tests.py coverage
```

This testing guide provides the foundation for maintaining high code quality in the BSP AI Assistant project. Follow these guidelines to write effective tests and maintain a robust testing suite.
