# Unit tests for app.py - Chainlit application main entry point
# Tests authentication, chat profiles, startup routines, and message processing

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Optional
import chainlit as cl
import sys
import os

# Add the parent directory to the path to import the app module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    header_auth_callback,
    chat_profile,
    set_starters,
    on_chat_resume,
    start,
    main
)


class TestHeaderAuthCallback:
    """Test cases for header_auth_callback function."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        self.valid_headers = {
            'X-MS-CLIENT-PRINCIPAL-NAME': 'test@microsoft.com',
            'X-MS-CLIENT-PRINCIPAL-ID': '1234567890'
        }
        self.empty_headers = {}
        self.partial_headers = {
            'X-MS-CLIENT-PRINCIPAL-NAME': 'test@microsoft.com'
        }

    def test_header_auth_callback_with_valid_headers(self):
        """Test authentication with valid headers."""
        result = header_auth_callback(self.valid_headers)
        
        assert result is not None
        assert isinstance(result, cl.User)
        assert result.identifier == 'test@microsoft.com'
        assert result.metadata['id'] == '1234567890'
        assert result.metadata['role'] == 'admin'
        assert result.metadata['provider'] == 'header'

    def test_header_auth_callback_with_empty_headers(self):
        """Test authentication with empty headers."""
        result = header_auth_callback(self.empty_headers)
        
        assert result is not None
        assert isinstance(result, cl.User)
        assert result.identifier == 'dummy@microsoft.com'
        assert result.metadata['id'] == '9876543210'

    def test_header_auth_callback_with_partial_headers(self):
        """Test authentication with partial headers."""
        result = header_auth_callback(self.partial_headers)
        
        assert result is not None
        assert isinstance(result, cl.User)
        assert result.identifier == 'test@microsoft.com'
        assert result.metadata['id'] == '9876543210'

    def test_header_auth_callback_with_none_user_name(self):
        """Test authentication when user name is None."""
        headers_with_none = {
            'X-MS-CLIENT-PRINCIPAL-NAME': None,
            'X-MS-CLIENT-PRINCIPAL-ID': '1234567890'
        }
        result = header_auth_callback(headers_with_none)
        
        assert result is None


class TestChatProfile:
    """Test cases for chat_profile function."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        self.mock_models = [
            {
                "model_deployment": "azure/gpt-4",
                "description": "Azure GPT-4 model for general conversations"
            },
            {
                "model_deployment": "foundry/gpt-4.1",
                "description": "Azure AI Foundry GPT-4.1 with advanced capabilities"
            },
            {
                "model_deployment": "perplexity/sonar",
                "description": "Perplexity Sonar model for research tasks"
            }
        ]

    @patch('app.get_llm_models')
    async def test_chat_profile_with_models(self, mock_get_llm_models):
        """Test chat profile creation with valid models."""
        mock_get_llm_models.return_value = self.mock_models
        
        profiles = await chat_profile()
        
        assert len(profiles) == 3
        assert all(isinstance(profile, cl.ChatProfile) for profile in profiles)
        
        # Check specific profile details
        azure_profile = next(p for p in profiles if p.name == "azure/gpt-4")
        assert azure_profile.markdown_description == "Azure GPT-4 model for general conversations"
        
        foundry_profile = next(p for p in profiles if p.name == "foundry/gpt-4.1")
        assert foundry_profile.markdown_description == "Azure AI Foundry GPT-4.1 with advanced capabilities"

    @patch('app.get_llm_models')
    async def test_chat_profile_with_empty_models(self, mock_get_llm_models):
        """Test chat profile creation with no models."""
        mock_get_llm_models.return_value = []
        
        profiles = await chat_profile()
        
        assert len(profiles) == 0
        assert isinstance(profiles, list)

    @patch('app.get_llm_models')
    async def test_chat_profile_with_malformed_model(self, mock_get_llm_models):
        """Test chat profile creation with malformed model data."""
        malformed_models = [
            {
                "model_deployment": "azure/gpt-4",
                # Missing description
            }
        ]
        mock_get_llm_models.return_value = malformed_models
        
        with pytest.raises(KeyError):
            await chat_profile()


class TestSetStarters:
    """Test cases for set_starters function."""
    
    async def test_set_starters_returns_correct_starters(self):
        """Test that set_starters returns the expected starter configurations."""
        starters = await set_starters()
        
        assert len(starters) == 4
        assert all(isinstance(starter, cl.Starter) for starter in starters)
        
        # Check specific starter details
        morning_starter = next(s for s in starters if s.label == "Morning routine ideation")
        assert morning_starter.icon == "/public/bulb.webp"
        assert "morning routine" in morning_starter.message.lower()
        
        error_starter = next(s for s in starters if s.label == "Spot the errors")
        assert error_starter.icon == "/public/warning.webp"
        
        productivity_starter = next(s for s in starters if s.label == "Get more done")
        assert productivity_starter.icon == "/public/rocket.png"
        
        knowledge_starter = next(s for s in starters if s.label == "Boost your knowledge")
        assert knowledge_starter.icon == "/public/book.png"

    async def test_set_starters_message_content(self):
        """Test that starter messages contain expected content."""
        starters = await set_starters()
        
        messages = [starter.message for starter in starters]
        
        # Check that messages are non-empty and meaningful
        assert all(len(message) > 10 for message in messages)
        assert any("morning routine" in message.lower() for message in messages)
        assert any("productivity" in message.lower() for message in messages)
        assert any("proofreading" in message.lower() for message in messages)


class TestOnChatResume:
    """Test cases for on_chat_resume function."""
    
    async def test_on_chat_resume_basic(self):
        """Test basic chat resume functionality."""
        mock_thread = Mock()
        mock_thread.id = "test-thread-123"
        
        # Should not raise any exceptions
        result = await on_chat_resume(mock_thread)
        assert result is None

    async def test_on_chat_resume_with_none_thread(self):
        """Test chat resume with None thread."""
        # Should handle None gracefully
        result = await on_chat_resume(None)
        assert result is None


class TestStart:
    """Test cases for start function."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        self.mock_settings = {
            "temperature": 0.7,
            "instructions": "Test instructions",
            "model_provider": "foundry"
        }

    @patch('app.cl.user_session')
    @patch('app.init_settings')
    @patch('app.get_llm_details')
    @patch('app.AgentsClient')
    @patch('app.DefaultAzureCredential')
    async def test_start_with_foundry_provider(self, mock_credential, mock_agents_client, 
                                              mock_get_llm_details, mock_init_settings, 
                                              mock_user_session):
        """Test start function with foundry provider."""
        # Mock setup
        mock_init_settings.return_value = self.mock_settings
        mock_get_llm_details.return_value = {
            "api_endpoint": "https://test-endpoint.com"
        }
        mock_user_session.get.side_effect = lambda key: {
            "chat_settings": self.mock_settings,
            "thread_id": None
        }.get(key)
        
        mock_thread = Mock()
        mock_thread.id = "test-thread-123"
        mock_client_instance = Mock()
        mock_client_instance.threads.create.return_value = mock_thread
        mock_agents_client.return_value = mock_client_instance
        
        # Execute function
        await start()
        
        # Verify calls
        mock_init_settings.assert_called_once()
        mock_get_llm_details.assert_called_once()
        mock_user_session.set.assert_called()
        mock_agents_client.assert_called_once()
        mock_client_instance.threads.create.assert_called_once()

    @patch('app.cl.user_session')
    @patch('app.init_settings')
    @patch('app.get_llm_details')
    async def test_start_with_non_foundry_provider(self, mock_get_llm_details, 
                                                  mock_init_settings, mock_user_session):
        """Test start function with non-foundry provider."""
        # Mock setup
        non_foundry_settings = {
            "temperature": 0.7,
            "instructions": "Test instructions",
            "model_provider": "azure"
        }
        mock_init_settings.return_value = non_foundry_settings
        mock_user_session.get.side_effect = lambda key: {
            "chat_settings": non_foundry_settings,
            "thread_id": None
        }.get(key)
        
        # Execute function
        await start()
        
        # Verify calls
        mock_init_settings.assert_called_once()
        mock_user_session.set.assert_called()

    @patch('app.cl.user_session')
    @patch('app.init_settings')
    @patch('app.cl.Message')
    async def test_start_with_exception(self, mock_message, mock_init_settings, mock_user_session):
        """Test start function when an exception occurs."""
        # Mock setup to raise exception
        mock_init_settings.side_effect = Exception("Test error")
        mock_message_instance = AsyncMock()
        mock_message.return_value = mock_message_instance
        
        # Execute function
        await start()
        
        # Verify error handling
        mock_message.assert_called_once()
        mock_message_instance.send.assert_called_once()


class TestMain:
    """Test cases for main function."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        self.mock_message = Mock()
        self.mock_message.content = "Test user message"
        self.mock_message.elements = []
        
        self.mock_settings = {
            "model_provider": "azure",
            "temperature": 0.7
        }

    @patch('app.cl.user_session')
    @patch('app.time.time')
    @patch('app.append_message')
    @patch('app.chat_completion')
    async def test_main_with_standard_provider(self, mock_chat_completion, mock_append_message,
                                              mock_time, mock_user_session):
        """Test main function with standard LLM provider."""
        # Mock setup
        mock_time.return_value = 1234567890
        mock_user_session.get.return_value = self.mock_settings
        mock_append_message.side_effect = [
            [{"role": "user", "content": "Test message"}],  # First call
            None  # Second call
        ]
        mock_chat_completion.return_value = "Test response"
        
        # Execute function
        await main(self.mock_message)
        
        # Verify calls
        mock_user_session.set.assert_called_with("start_time", 1234567890)
        assert mock_append_message.call_count == 2
        mock_chat_completion.assert_called_once()

    @patch('app.cl.user_session')
    @patch('app.time.time')
    @patch('app.append_message')
    @patch('app.chat_agent')
    async def test_main_with_foundry_provider(self, mock_chat_agent, mock_append_message,
                                             mock_time, mock_user_session):
        """Test main function with foundry provider."""
        # Mock setup
        foundry_settings = {"model_provider": "foundry"}
        mock_time.return_value = 1234567890
        mock_user_session.get.return_value = foundry_settings
        mock_append_message.side_effect = [
            [{"role": "user", "content": "Test message"}],  # First call
            None  # Second call
        ]
        mock_chat_agent.return_value = "Test foundry response"
        
        # Execute function
        await main(self.mock_message)
        
        # Verify calls
        mock_user_session.set.assert_called_with("start_time", 1234567890)
        assert mock_append_message.call_count == 2
        mock_chat_agent.assert_called_once_with("Test user message")

    @patch('app.cl.user_session')
    @patch('app.time.time')
    @patch('app.append_message')
    @patch('app.cl.Message')
    async def test_main_with_exception(self, mock_message_class, mock_append_message,
                                      mock_time, mock_user_session):
        """Test main function when an exception occurs."""
        # Mock setup to raise exception
        mock_time.return_value = 1234567890
        mock_user_session.get.return_value = self.mock_settings
        mock_append_message.side_effect = Exception("Test error")
        
        mock_error_message = AsyncMock()
        mock_message_class.return_value = mock_error_message
        
        # Execute function
        await main(self.mock_message)
        
        # Verify error handling
        mock_message_class.assert_called_once()
        mock_error_message.send.assert_called_once()

    @patch('app.cl.user_session')
    @patch('app.time.time')
    @patch('app.append_message')
    @patch('app.chat_completion')
    async def test_main_with_file_elements(self, mock_chat_completion, mock_append_message,
                                          mock_time, mock_user_session):
        """Test main function with file attachments."""
        # Mock setup with file elements
        mock_element = Mock()
        mock_element.name = "test.txt"
        mock_element.path = "/path/to/test.txt"
        
        message_with_files = Mock()
        message_with_files.content = "Test message with files"
        message_with_files.elements = [mock_element]
        
        mock_time.return_value = 1234567890
        mock_user_session.get.return_value = self.mock_settings
        mock_append_message.side_effect = [
            [{"role": "user", "content": "Test message"}],  # First call
            None  # Second call
        ]
        mock_chat_completion.return_value = "Test response"
        
        # Execute function
        await main(message_with_files)
        
        # Verify calls
        mock_append_message.assert_called()
        # First call should include the elements
        first_call_args = mock_append_message.call_args_list[0]
        assert first_call_args[0][0] == "user"  # role
        assert first_call_args[0][1] == "Test message with files"  # content
        assert first_call_args[0][2] == [mock_element]  # elements


if __name__ == "__main__":
    pytest.main([__file__])
