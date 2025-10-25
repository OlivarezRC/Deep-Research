#!/usr/bin/env python
"""
Test runner script for BSP AI Assistant
Provides comprehensive testing functionality with coverage reporting and organized test execution.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """
    Run a command and handle errors gracefully.
    
    Args:
        command: List of command arguments
        description: Description of what the command does
        
    Returns:
        bool: True if command succeeded, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {command[0]}")
        return False


def install_dependencies():
    """Install testing dependencies."""
    print("Installing testing dependencies...")
    return run_command([
        sys.executable, "-m", "pip", "install", 
        "pytest", "pytest-asyncio", "pytest-mock", "pytest-cov"
    ], "Installing testing dependencies")


def run_unit_tests():
    """Run all unit tests."""
    return run_command([
        sys.executable, "-m", "pytest", 
        "tests/", 
        "-v", 
        "--tb=short",
        "-x"  # Stop on first failure
    ], "Running unit tests")


def run_tests_with_coverage():
    """Run tests with coverage reporting."""
    return run_command([
        sys.executable, "-m", "pytest",
        "tests/",
        "--cov=app",
        "--cov=utils",
        "--cov-report=html",
        "--cov-report=term-missing",
        "-v"
    ], "Running tests with coverage")


def run_specific_test_file(test_file):
    """Run a specific test file."""
    if not test_file.startswith("test_"):
        test_file = f"test_{test_file}"
    if not test_file.endswith(".py"):
        test_file = f"{test_file}.py"
    
    test_path = Path("tests") / test_file
    if not test_path.exists():
        print(f"❌ Test file not found: {test_path}")
        return False
    
    return run_command([
        sys.executable, "-m", "pytest",
        str(test_path),
        "-v",
        "--tb=short"
    ], f"Running specific test file: {test_file}")


def run_linting():
    """Run code linting with flake8."""
    return run_command([
        sys.executable, "-m", "flake8",
        "app.py", "utils/", "tests/",
        "--max-line-length=120",
        "--ignore=E501,W503"
    ], "Running code linting")


def check_test_discovery():
    """Check test discovery without running tests."""
    return run_command([
        sys.executable, "-m", "pytest",
        "--collect-only",
        "tests/"
    ], "Checking test discovery")


def main():
    """Main test runner interface."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "install":
            install_dependencies()
        elif command == "unit":
            run_unit_tests()
        elif command == "coverage":
            run_tests_with_coverage()
        elif command == "lint":
            run_linting()
        elif command == "discover":
            check_test_discovery()
        elif command == "file" and len(sys.argv) > 2:
            test_file = sys.argv[2]
            run_specific_test_file(test_file)
        elif command == "all":
            print("Running comprehensive test suite...")
            success = True
            success &= install_dependencies()
            success &= check_test_discovery()
            success &= run_unit_tests()
            success &= run_tests_with_coverage()
            
            if success:
                print(f"\n{'='*60}")
                print("🎉 All tests completed successfully!")
                print(f"{'='*60}")
            else:
                print(f"\n{'='*60}")
                print("❌ Some tests failed. Please check the output above.")
                print(f"{'='*60}")
                sys.exit(1)
        else:
            print_usage()
    else:
        print_usage()


def print_usage():
    """Print usage information."""
    print("""
BSP AI Assistant Test Runner

Usage:
    python run_tests.py <command> [options]

Commands:
    install     - Install testing dependencies
    unit        - Run unit tests only
    coverage    - Run tests with coverage reporting
    lint        - Run code linting
    discover    - Check test discovery
    file <name> - Run specific test file (e.g., 'file app' or 'file test_app.py')
    all         - Run comprehensive test suite

Examples:
    python run_tests.py install
    python run_tests.py unit
    python run_tests.py coverage
    python run_tests.py file app
    python run_tests.py all

For pytest-specific options, run pytest directly:
    python -m pytest tests/ -v --tb=short
    """)


if __name__ == "__main__":
    main()
