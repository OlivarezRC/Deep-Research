# Unit tests for utils/utils.py - Core utility functions
# Tests session management, logging, message formatting, model configuration, and chat settings

import pytest
import os
import json
import base64
import tempfile
from unittest.mock import Mock, patch, MagicMock, mock_open, AsyncMock
import chainlit as cl
import sys

# Add the parent directory to the path to import the utils module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import (
    truncate,
    add_context,
    get_logger,
    get_llm_models,
    append_message,
    init_settings,
    get_llm_details
)


class TestTruncate:
    """Test cases for truncate function."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        self.short_record = {"message": "Short message"}
        self.long_record = {"message": "x" * 1500}  # Message longer than 1000 chars
        self.exact_limit_record = {"message": "x" * 1000}  # Exactly 1000 chars

    def test_truncate_short_message(self):
        """Test truncate with message shorter than limit."""
        result = truncate(self.short_record)
        
        assert result is True
        assert self.short_record["message"] == "Short message"

    def test_truncate_long_message(self):
        """Test truncate with message longer than limit."""
        original_length = len(self.long_record["message"])
        result = truncate(self.long_record)
        
        assert result is True
        assert len(self.long_record["message"]) < original_length
        assert self.long_record["message"].endswith("… [truncated]")
        assert len(self.long_record["message"]) == 1013  # 1000 + len("… [truncated]")

    def test_truncate_exact_limit_message(self):
        """Test truncate with message exactly at limit."""
        result = truncate(self.exact_limit_record)
        
        assert result is True
        assert self.exact_limit_record["message"] == "x" * 1000
        assert not self.exact_limit_record["message"].endswith("… [truncated]")

    def test_truncate_empty_message(self):
        """Test truncate with empty message."""
        empty_record = {"message": ""}
        result = truncate(empty_record)
        
        assert result is True
        assert empty_record["message"] == ""


class TestAddContext:
    """Test cases for add_context function."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        self.mock_record = {"extra": {}}

    @patch('utils.utils.cl.user_session')
    def test_add_context_with_full_user_data(self, mock_user_session):
        """Test add_context with complete user information."""
        # Mock user with metadata
        mock_user = Mock()
        mock_user.metadata = {"id": "user123"}
        mock_user.identifier = "test@example.com"
        
        mock_user_session.get.side_effect = lambda key, default=None: {
            "id": "session123",
            "user": mock_user
        }.get(key, default)
        
        result = add_context(self.mock_record)
        
        assert result is True
        assert self.mock_record["extra"]["session_id"] == "session123"
        assert self.mock_record["extra"]["user_id"] == "user123"

    @patch('utils.utils.cl.user_session')
    def test_add_context_with_user_identifier_only(self, mock_user_session):
        """Test add_context with user having only identifier."""
        # Mock user without metadata but with identifier
        mock_user = Mock()
        mock_user.metadata = {}
        mock_user.identifier = "test@example.com"
        
        mock_user_session.get.side_effect = lambda key, default=None: {
            "id": "session123",
            "user": mock_user
        }.get(key, default)
        
        result = add_context(self.mock_record)
        
        assert result is True
        assert self.mock_record["extra"]["session_id"] == "session123"
        assert self.mock_record["extra"]["user_id"] == "test@example.com"

    @patch('utils.utils.cl.user_session')
    def test_add_context_with_no_user(self, mock_user_session):
        """Test add_context with no user information."""
        mock_user_session.get.side_effect = lambda key, default=None: {
            "id": "session123",
            "user": None
        }.get(key, default)
        
        result = add_context(self.mock_record)
        
        assert result is True
        assert self.mock_record["extra"]["session_id"] == "session123"
        assert self.mock_record["extra"]["user_id"] == "anonymous"

    @patch('utils.utils.cl.user_session')
    def test_add_context_with_unknown_session(self, mock_user_session):
        """Test add_context with unknown session."""
        mock_user_session.get.side_effect = lambda key, default=None: {
            "user": None
        }.get(key, default)  # Note: "id" key is missing, so default will be used
        
        result = add_context(self.mock_record)
        
        assert result is True
        assert self.mock_record["extra"]["session_id"] == "unknown-session"
        assert self.mock_record["extra"]["user_id"] == "anonymous"


class TestGetLogger:
    """Test cases for get_logger function."""
    
    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logger instance."""
        logger = get_logger()
        
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'debug')
        assert hasattr(logger, 'warning')


class TestGetLlmModels:
    """Test cases for get_llm_models function."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        self.sample_config = [
            {
                "model_deployment": "azure/gpt-4",
                "description": "Azure GPT-4 model",
                "api_key": "test-key-1"
            },
            {
                "model_deployment": "foundry/gpt-4.1",
                "description": "Foundry GPT-4.1 model",
                "api_key": "test-key-2"
            }
        ]

    @patch.dict('os.environ', {'LLM_CONFIG': '[]'})
    def test_get_llm_models_from_env_empty(self):
        """Test get_llm_models with empty environment variable."""
        result = get_llm_models()
        
        assert result == []
        assert isinstance(result, list)

    @patch.dict('os.environ', {'LLM_CONFIG': ''}, clear=True)
    @patch('builtins.open', mock_open(read_data='[]'))
    def test_get_llm_models_from_file_when_no_env(self):
        """Test get_llm_models falls back to file when no env var."""
        with patch('utils.utils.json.loads') as mock_json_loads:
            mock_json_loads.side_effect = Exception("No env var")
            
            with patch('utils.utils.json.load') as mock_json_load:
                mock_json_load.return_value = self.sample_config
                
                result = get_llm_models()
                
                assert result == self.sample_config

    @patch.dict('os.environ', {'LLM_CONFIG': '[{"model_deployment": "test/model"}]'})
    def test_get_llm_models_from_env_valid_json(self):
        """Test get_llm_models with valid JSON in environment variable."""
        result = get_llm_models()
        
        assert len(result) == 1
        assert result[0]["model_deployment"] == "test/model"

    @patch.dict('os.environ', {'LLM_CONFIG': 'invalid-json'})
    @patch('builtins.open', new_callable=mock_open, read_data='[{"model_deployment": "fallback/model", "model_provider": "test"}]')
    def test_get_llm_models_from_env_invalid_json(self, mock_file):
        """Test get_llm_models falls back to file when environment variable has invalid JSON."""
        result = get_llm_models()
        
        # Should fall back to file reading
        mock_file.assert_called_once_with("llm_config/llm_config.json", "r")
        assert len(result) == 1
        assert result[0]["model_deployment"] == "fallback/model"


class TestAppendMessage:
    """Test cases for append_message function."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        self.mock_settings = {
            "instructions": "You are a helpful AI assistant.",
            "model_provider": "azure"
        }
        
        self.mock_image_element = Mock()
        self.mock_image_element.mime = "image/png"
        self.mock_image_element.path = "/path/to/image.png"
        self.mock_image_element.name = "image.png"
        
        self.mock_text_element = Mock()
        self.mock_text_element.mime = "text/plain"
        self.mock_text_element.path = "test_file.txt"  # Use relative path to test file
        self.mock_text_element.name = "test_file.txt"

    @patch('utils.utils.cl.user_session')
    def test_append_message_user_basic(self, mock_user_session):
        """Test append_message for basic user message."""
        mock_user_session.get.side_effect = lambda key, default=None: {
            "chat_settings": self.mock_settings,
            "chat_history": []
        }.get(key, default)
        
        result = append_message("user", "Hello world")
        
        assert len(result) == 2  # System prompt + user message
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[1]["content"][0]["text"] == "Hello world"

    @patch('utils.utils.cl.user_session')
    def test_append_message_assistant_basic(self, mock_user_session):
        """Test append_message for basic assistant message."""
        mock_user_session.get.side_effect = lambda key, default=None: {
            "chat_settings": self.mock_settings,
            "chat_history": []
        }.get(key, default)
        
        result = append_message("assistant", "Hello! How can I help you?")
        
        assert len(result) == 2  # System prompt + assistant message
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"][0]["text"] == "Hello! How can I help you?"

    @patch('utils.utils.cl.user_session')
    @patch('builtins.open', mock_open(read_data=b'fake_image_data'))
    @patch('utils.utils.base64.b64encode')
    def test_append_message_with_image_element(self, mock_b64_encode, mock_user_session):
        """Test append_message with image element."""
        mock_b64_encode.return_value = b'encoded_image_data'
        mock_user_session.get.side_effect = lambda key, default=None: {
            "chat_settings": self.mock_settings,
            "chat_history": []
        }.get(key, default)
        
        result = append_message("user", "Look at this image", [self.mock_image_element])
        
        assert len(result) == 2
        assert result[1]["role"] == "user"
        assert len(result[1]["content"]) == 2  # Text + image
        assert result[1]["content"][0]["text"] == "Look at this image"
        assert result[1]["content"][1]["type"] == "image_url"

    @patch('utils.utils.cl.user_session')
    @patch('utils.utils.md.convert')
    def test_append_message_with_text_element(self, mock_md_convert, mock_user_session):
        """Test append_message with text file element."""
        mock_result = Mock()
        mock_result.text_content = "File content here"
        mock_md_convert.return_value = mock_result
        
        mock_user_session.get.side_effect = lambda key, default=None: {
            "chat_settings": self.mock_settings,
            "chat_history": []
        }.get(key, default)
        
        result = append_message("user", "Check this file", [self.mock_text_element])
        
        assert len(result) == 2
        assert result[1]["role"] == "user"
        assert len(result[1]["content"]) == 2  # Text + file content
        assert "<file_name:test_file.txt>" in result[1]["content"][1]["text"]

    @patch('utils.utils.cl.user_session')
    def test_append_message_with_foundry_provider(self, mock_user_session):
        """Test append_message with foundry provider."""
        foundry_settings = self.mock_settings.copy()
        foundry_settings["model_provider"] = "foundry"
        
        mock_user_session.get.side_effect = lambda key, default=None: {
            "chat_settings": foundry_settings,
            "chat_history": []
        }.get(key, default)
        
        result = append_message("user", "Test message", [self.mock_text_element])
        
        # With foundry, files should be uploaded, not converted to markdown
        assert len(result) == 2
        assert result[1]["role"] == "user"
        mock_user_session.set.assert_called()

    @patch('utils.utils.cl.user_session')
    def test_append_message_chat_history_pruning(self, mock_user_session):
        """Test that chat history is pruned to 10 messages."""
        # Create a long chat history (15 messages)
        long_history = []
        for i in range(15):
            long_history.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": [{"type": "text", "text": f"Message {i}"}]
            })
        
        mock_user_session.get.side_effect = lambda key, default=None: {
            "chat_settings": self.mock_settings,
            "chat_history": long_history
        }.get(key, default)
        
        result = append_message("assistant", "New response")
        
        # Should have system prompt + 10 most recent messages
        assert len(result) == 11
        # Verify pruning happened by checking the saved history
        calls = mock_user_session.set.call_args_list
        history_call = next(call for call in calls if call[0][0] == "chat_history")
        saved_history = history_call[0][1]
        assert len(saved_history) == 10

    @patch('utils.utils.cl.user_session')
    def test_append_message_no_pruning_for_user(self, mock_user_session):
        """Test that chat history is not pruned for user messages."""
        # Create a long chat history (15 messages)
        long_history = []
        for i in range(15):
            long_history.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": [{"type": "text", "text": f"Message {i}"}]
            })
        
        mock_user_session.get.side_effect = lambda key, default=None: {
            "chat_settings": self.mock_settings,
            "chat_history": long_history
        }.get(key, default)
        
        result = append_message("user", "New user message")
        
        # Should have system prompt + all 16 messages (15 + new one)
        assert len(result) == 17


class TestInitSettings:
    """Test cases for init_settings function."""
    
    @patch('utils.utils.cl.ChatSettings')
    async def test_init_settings_returns_settings(self, mock_chat_settings):
        """Test that init_settings returns expected settings."""
        mock_settings_instance = AsyncMock()
        mock_settings_instance.send.return_value = {
            "temperature": 0.7,
            "instructions": "Test instructions"
        }
        mock_chat_settings.return_value = mock_settings_instance
        
        result = await init_settings()
        
        assert result is not None
        mock_chat_settings.assert_called_once()
        mock_settings_instance.send.assert_called_once()

    @patch('utils.utils.cl.ChatSettings')
    async def test_init_settings_temperature_slider(self, mock_chat_settings):
        """Test that temperature slider is configured correctly."""
        mock_settings_instance = AsyncMock()
        mock_chat_settings.return_value = mock_settings_instance
        
        await init_settings()
        
        # Check that ChatSettings was called with correct arguments
        call_args = mock_chat_settings.call_args[0][0]
        
        # Find temperature slider
        temp_slider = next(widget for widget in call_args if widget.id == "temperature")
        assert temp_slider.label == "Temperature"
        assert temp_slider.initial == 0.7
        assert temp_slider.min == 0
        assert temp_slider.max == 2
        assert temp_slider.step == 0.1

    @patch('utils.utils.cl.ChatSettings')
    async def test_init_settings_instructions_input(self, mock_chat_settings):
        """Test that instructions input is configured correctly."""
        mock_settings_instance = AsyncMock()
        mock_chat_settings.return_value = mock_settings_instance
        
        await init_settings()
        
        # Check that ChatSettings was called with correct arguments
        call_args = mock_chat_settings.call_args[0][0]
        
        # Find instructions input
        instructions_input = next(widget for widget in call_args if widget.id == "instructions")
        assert instructions_input.label == "Instructions"
        assert "BSP AI Assistant" in instructions_input.initial
        assert "Bangko Sentral ng Pilipinas" in instructions_input.initial


class TestGetLlmDetails:
    """Test cases for get_llm_details function."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        self.mock_models = [
            {
                "model_deployment": "azure/gpt-4",
                "api_key": "test-key-1",
                "api_endpoint": "https://test1.openai.azure.com"
            },
            {
                "model_deployment": "foundry/gpt-4.1",
                "api_key": "test-key-2",
                "api_endpoint": "https://test2.foundry.azure.com"
            }
        ]

    @patch('utils.utils.cl.user_session')
    @patch('utils.utils.get_llm_models')
    def test_get_llm_details_azure_model(self, mock_get_llm_models, mock_user_session):
        """Test get_llm_details for Azure model."""
        mock_get_llm_models.return_value = self.mock_models
        
        mock_chat_settings = {"temperature": 0.7}
        mock_user_session.get.side_effect = lambda key, default=None: {
            "chat_settings": mock_chat_settings,
            "chat_profile": "azure/gpt-4"
        }.get(key, default)
        
        result = get_llm_details()
        
        assert result["model_deployment"] == "azure/gpt-4"
        assert result["api_key"] == "test-key-1"
        assert result["api_endpoint"] == "https://test1.openai.azure.com"
        
        # Verify session was updated
        mock_user_session.set.assert_called_once()
        set_call = mock_user_session.set.call_args
        assert set_call[0][0] == "chat_settings"
        updated_settings = set_call[0][1]
        assert updated_settings["model_name"] == "gpt-4"
        assert updated_settings["model_provider"] == "azure"

    @patch('utils.utils.cl.user_session')
    @patch('utils.utils.get_llm_models')
    def test_get_llm_details_foundry_model(self, mock_get_llm_models, mock_user_session):
        """Test get_llm_details for Foundry model."""
        mock_get_llm_models.return_value = self.mock_models
        
        mock_chat_settings = {"temperature": 0.7}
        mock_user_session.get.side_effect = lambda key, default=None: {
            "chat_settings": mock_chat_settings,
            "chat_profile": "foundry/gpt-4.1"
        }.get(key, default)
        
        result = get_llm_details()
        
        assert result["model_deployment"] == "foundry/gpt-4.1"
        assert result["api_key"] == "test-key-2"
        assert result["api_endpoint"] == "https://test2.foundry.azure.com"
        
        # Verify session was updated with correct provider and model
        mock_user_session.set.assert_called_once()
        set_call = mock_user_session.set.call_args
        updated_settings = set_call[0][1]
        assert updated_settings["model_name"] == "gpt-4.1"
        assert updated_settings["model_provider"] == "foundry"

    @patch('utils.utils.cl.user_session')
    @patch('utils.utils.get_llm_models')
    def test_get_llm_details_model_not_found(self, mock_get_llm_models, mock_user_session):
        """Test get_llm_details when model is not found."""
        mock_get_llm_models.return_value = self.mock_models
        
        mock_chat_settings = {"temperature": 0.7}
        mock_user_session.get.side_effect = lambda key, default=None: {
            "chat_settings": mock_chat_settings,
            "chat_profile": "nonexistent/model"
        }.get(key, default)
        
        result = get_llm_details()
        
        assert result == {}  # Should return empty dict when not found

    @patch('utils.utils.cl.user_session')
    @patch('utils.utils.get_llm_models')
    def test_get_llm_details_empty_models_list(self, mock_get_llm_models, mock_user_session):
        """Test get_llm_details with empty models list."""
        mock_get_llm_models.return_value = []
        
        mock_chat_settings = {"temperature": 0.7}
        mock_user_session.get.side_effect = lambda key, default=None: {
            "chat_settings": mock_chat_settings,
            "chat_profile": "azure/gpt-4"
        }.get(key, default)
        
        result = get_llm_details()
        
        assert result == {}  # Should return empty dict when no models


if __name__ == "__main__":
    pytest.main([__file__])
