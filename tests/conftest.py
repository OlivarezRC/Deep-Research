# Shared test configuration and utilities for BSP AI Assistant tests
# This file contains common test data, fixtures, and helper functions

import pytest
import tempfile
import json
import os
from unittest.mock import Mock, AsyncMock
from pathlib import Path


# ============================================================================
# SHARED TEST DATA
# ============================================================================

# Sample LLM model configurations for testing
SAMPLE_LLM_MODELS = [
    {
        "model_deployment": "azure/gpt-4",
        "description": "Azure GPT-4 model for general conversations",
        "api_key": "test-azure-key-1",
        "api_endpoint": "https://test1.openai.azure.com",
        "api_version": "2023-05-15"
    },
    {
        "model_deployment": "azure/gpt-3.5-turbo",
        "description": "Azure GPT-3.5 Turbo for quick responses",
        "api_key": "test-azure-key-2",
        "api_endpoint": "https://test2.openai.azure.com",
        "api_version": "2023-05-15"
    },
    {
        "model_deployment": "foundry/gpt-4.1",
        "description": "Azure AI Foundry GPT-4.1 with advanced capabilities",
        "api_key": "test-foundry-key",
        "api_endpoint": "https://test.foundry.azure.com",
        "model_id": "asst_test123"
    },
    {
        "model_deployment": "perplexity/sonar",
        "description": "Perplexity Sonar model for research tasks",
        "api_key": "test-perplexity-key"
    },
    {
        "model_deployment": "azure/o3-mini",
        "description": "Azure O3-Mini model (no temperature)",
        "api_key": "test-o3-key",
        "api_endpoint": "https://test3.openai.azure.com",
        "api_version": "2023-05-15"
    }
]

# Sample chat messages for testing
SAMPLE_CHAT_MESSAGES = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful AI assistant for BSP employees."}]
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "Hello, can you help me with BSP policies?"}]
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "Of course! I'm here to help with BSP policies and procedures."}]
    }
]

# Sample chat settings
SAMPLE_CHAT_SETTINGS = {
    "temperature": 0.7,
    "instructions": "You are BSP AI Assistant, an advanced conversational AI model...",
    "model_provider": "azure",
    "model_name": "gpt-4"
}

# Sample user session data
SAMPLE_USER_SESSION = {
    "id": "test-session-123",
    "user": Mock(identifier="test@bsp.gov.ph", metadata={"id": "user123", "role": "admin"}),
    "chat_settings": SAMPLE_CHAT_SETTINGS,
    "chat_profile": "azure/gpt-4",
    "chat_history": SAMPLE_CHAT_MESSAGES[1:],  # Exclude system message
    "thread_id": "thread-abc123",
    "uploaded_files": [],
    "start_time": 1234567880
}

# Sample file elements for testing file uploads
SAMPLE_IMAGE_ELEMENT = Mock(
    mime="image/png",
    path="/path/to/test_image.png",
    name="test_image.png"
)

SAMPLE_TEXT_ELEMENT = Mock(
    mime="text/plain",
    path="/path/to/test_document.txt",
    name="test_document.txt"
)

SAMPLE_PDF_ELEMENT = Mock(
    mime="application/pdf",
    path="/path/to/test_document.pdf",
    name="test_document.pdf"
)


# ============================================================================
# PYTEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_llm_models():
    """Fixture providing sample LLM model configurations."""
    return SAMPLE_LLM_MODELS.copy()


@pytest.fixture
def sample_chat_messages():
    """Fixture providing sample chat messages."""
    return [msg.copy() for msg in SAMPLE_CHAT_MESSAGES]


@pytest.fixture
def sample_chat_settings():
    """Fixture providing sample chat settings."""
    return SAMPLE_CHAT_SETTINGS.copy()


@pytest.fixture
def sample_user_session():
    """Fixture providing sample user session data."""
    return SAMPLE_USER_SESSION.copy()


@pytest.fixture
def sample_file_elements():
    """Fixture providing sample file elements."""
    return {
        "image": SAMPLE_IMAGE_ELEMENT,
        "text": SAMPLE_TEXT_ELEMENT,
        "pdf": SAMPLE_PDF_ELEMENT
    }


@pytest.fixture
def temp_config_file():
    """Fixture providing a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(SAMPLE_LLM_MODELS, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def mock_chainlit_session():
    """Fixture providing a mocked Chainlit user session."""
    with pytest.mock.patch('chainlit.user_session') as mock_session:
        mock_session.get.side_effect = lambda key: SAMPLE_USER_SESSION.get(key)
        mock_session.set = Mock()
        yield mock_session


@pytest.fixture(autouse=True)
def disable_loguru_handlers():
    """Disable loguru handlers during tests to avoid logging errors."""
    from loguru import logger
    logger.remove()  # Remove all handlers
    # Add a simple console handler for test output
    logger.add(
        lambda message: None,  # No-op handler
        level="DEBUG",
        filter=lambda record: True  # Simple filter that always returns True
    )
    yield
    logger.remove()  # Clean up after test


@pytest.fixture
def mock_chainlit_message():
    """Fixture providing a mocked Chainlit message."""
    mock_message = AsyncMock()
    mock_message.content = ""
    mock_message.elements = []
    mock_message.send = AsyncMock()
    mock_message.update = AsyncMock()
    return mock_message


@pytest.fixture
def mock_logger():
    """Fixture providing a mocked logger."""
    with pytest.mock.patch('utils.utils.logger') as mock_log:
        yield mock_log


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_mock_llm_response_chunk(content: str, has_citations: bool = False):
    """
    Create a mock LLM response chunk for testing.
    
    Args:
        content: The content for the chunk
        has_citations: Whether to include citations
        
    Returns:
        Mock: A mock response chunk
    """
    chunk = Mock()
    chunk.choices = [Mock()]
    chunk.choices[0].delta.content = content
    
    if has_citations:
        chunk.citations = [
            "https://example.com/source1",
            "https://example.com/source2"
        ]
    
    return chunk


def create_mock_azure_ai_agent_response(text: str, annotations: list = None):
    """
    Create a mock Azure AI agent response for testing.
    
    Args:
        text: The response text
        annotations: List of annotation objects
        
    Returns:
        Mock: A mock agent response
    """
    response = Mock()
    response.text.value = text
    response.text.annotations = annotations or []
    return response


def create_mock_file_upload_element(file_type: str = "text", file_name: str = None):
    """
    Create a mock file upload element for testing.
    
    Args:
        file_type: Type of file ('text', 'image', 'pdf')
        file_name: Optional custom file name
        
    Returns:
        Mock: A mock file element
    """
    type_mapping = {
        "text": ("text/plain", "test.txt"),
        "image": ("image/png", "test.png"),
        "pdf": ("application/pdf", "test.pdf")
    }
    
    mime_type, default_name = type_mapping.get(file_type, ("text/plain", "test.txt"))
    final_name = file_name or default_name
    
    element = Mock()
    element.mime = mime_type
    element.name = final_name
    element.path = f"/path/to/{final_name}"
    
    return element


def assert_message_structure(message: dict, expected_role: str):
    """
    Assert that a message has the expected structure.
    
    Args:
        message: The message dictionary to check
        expected_role: The expected role ('user', 'assistant', 'system')
    """
    assert isinstance(message, dict)
    assert "role" in message
    assert "content" in message
    assert message["role"] == expected_role
    assert isinstance(message["content"], list)
    
    # Check that content items have proper structure
    for content_item in message["content"]:
        assert isinstance(content_item, dict)
        assert "type" in content_item
        if content_item["type"] == "text":
            assert "text" in content_item
        elif content_item["type"] == "image_url":
            assert "image_url" in content_item


def assert_llm_params_structure(params: dict):
    """
    Assert that LLM parameters have the expected structure.
    
    Args:
        params: The parameters dictionary to check
    """
    required_fields = ["model", "messages", "stream", "api_key"]
    for field in required_fields:
        assert field in params, f"Missing required field: {field}"
    
    assert isinstance(params["messages"], list)
    assert isinstance(params["stream"], bool)
    assert isinstance(params["api_key"], str)


def create_temp_file_with_content(content: str, suffix: str = ".txt"):
    """
    Create a temporary file with specific content.
    
    Args:
        content: Content to write to the file
        suffix: File suffix/extension
        
    Returns:
        str: Path to the temporary file
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        f.write(content)
        return f.name


def cleanup_temp_file(file_path: str):
    """
    Clean up a temporary file.
    
    Args:
        file_path: Path to the file to delete
    """
    try:
        os.unlink(file_path)
    except (FileNotFoundError, OSError):
        pass


# ============================================================================
# TEST MARKERS
# ============================================================================

# Custom pytest markers for organizing tests
pytest_markers = {
    "unit": "Unit tests that test individual functions in isolation",
    "integration": "Integration tests that test multiple components together",
    "slow": "Tests that take a long time to run",
    "requires_api": "Tests that require external API access",
    "requires_files": "Tests that require file system operations"
}
