# Unit tests for utils/foundry.py - Azure AI Foundry agent integration
# Tests Azure AI agent interactions, file processing, and streaming responses

import pytest
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import chainlit as cl
import sys
import os

# Add the parent directory to the path to import the foundry module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.foundry import chat_agent


class TestChatAgent:
    """Test cases for chat_agent function."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        self.mock_llm_details = {
            "model_deployment": "foundry/gpt-4.1",
            "api_endpoint": "https://test.foundry.azure.com",
            "model_id": "asst_test123"
        }
        
        self.mock_settings = {
            "model_name": "gpt-4.1",
            "temperature": 0.7
        }

    @patch('utils.foundry.cl.user_session')
    @patch('utils.foundry.cl.Message')
    @patch('utils.foundry.get_llm_models')
    @patch('utils.foundry.AgentsClient')
    @patch('utils.foundry.DefaultAzureCredential')
    @patch('utils.foundry.time.time')
    async def test_chat_agent_basic_response(self, mock_time, mock_credential, mock_agents_client, 
                                           mock_get_llm_models, mock_message_class, mock_user_session):
        """Test basic chat agent response without files."""
        # Mock setup
        mock_time.return_value = 1234567890
        mock_user_session.get.side_effect = lambda key, default=None: {
            "chat_settings": self.mock_settings,
            "chat_profile": "foundry/gpt-4.1",
            "thread_id": "thread123",
            "file_uploads": [],
            "file_contents": [],
            "uploaded_files": [],
            "start_time": 1234567880
        }.get(key, default)
        
        mock_message_instance = AsyncMock()
        mock_message_instance.content = ""
        mock_message_class.return_value = mock_message_instance
        
        mock_get_llm_models.return_value = [self.mock_llm_details]
        
        # Mock AgentsClient
        mock_client_instance = Mock()
        mock_agents_client.return_value = mock_client_instance
        
        # Mock stream events
        from azure.ai.agents.models import MessageDeltaChunk, ThreadRun, AgentStreamEvent
        
        mock_delta_chunk1 = Mock(spec=MessageDeltaChunk)
        mock_delta_chunk1.text = "Hello"
        
        mock_delta_chunk2 = Mock(spec=MessageDeltaChunk)
        mock_delta_chunk2.text = " world!"
        
        mock_thread_run = Mock(spec=ThreadRun)
        mock_thread_run.status = "completed"
        
        mock_stream_events = [
            (AgentStreamEvent.THREAD_MESSAGE_DELTA, mock_delta_chunk1, None),
            (AgentStreamEvent.THREAD_MESSAGE_DELTA, mock_delta_chunk2, None),
            (AgentStreamEvent.THREAD_RUN_COMPLETED, mock_thread_run, None)
        ]
        
        mock_stream_context = MagicMock()
        mock_stream_context.__enter__.return_value = mock_stream_events
        mock_stream_context.__exit__.return_value = None
        mock_client_instance.runs.stream.return_value = mock_stream_context
        
        # Mock get_last_message_text_by_role
        mock_response_message = Mock()
        mock_response_message.text.value = "Hello world!"
        mock_response_message.text.annotations = []
        mock_client_instance.messages.get_last_message_text_by_role.return_value = mock_response_message
        
        # Mock messages.list for image processing
        mock_client_instance.messages.list.return_value = []
        
        # Execute function
        result = await chat_agent("Test message")
        
        # Verify result
        assert result == "Hello world!"
        mock_client_instance.messages.create.assert_called_once()
        mock_client_instance.runs.stream.assert_called_once()

    @patch('utils.foundry.cl.user_session')
    @patch('utils.foundry.cl.Message')
    @patch('utils.foundry.get_llm_models')
    @patch('utils.foundry.AgentsClient')
    @patch('utils.foundry.DefaultAzureCredential')
    async def test_chat_agent_with_file_upload(self, mock_credential, mock_agents_client, 
                                              mock_get_llm_models, mock_message_class, mock_user_session):
        """Test chat agent with file upload."""
        # Mock setup
        mock_user_session.get.side_effect = lambda key, default=None: {
            "chat_settings": self.mock_settings,
            "chat_profile": "foundry/gpt-4.1",
            "thread_id": "thread123",
            "file_uploads": [{"name": "test.txt", "mime": "text/plain", "path": "/path/to/test.txt", "base64": None}],
            "file_contents": [],
            "uploaded_files": ["/path/to/test.txt"],
            "start_time": 1234567880
        }.get(key, default)
        
        mock_message_instance = AsyncMock()
        mock_message_instance.content = ""
        mock_message_class.return_value = mock_message_instance
        
        mock_get_llm_models.return_value = [self.mock_llm_details]
        
        # Mock AgentsClient
        mock_client_instance = Mock()
        mock_agents_client.return_value = mock_client_instance
        
        # Mock file upload
        mock_uploaded_file = Mock()
        mock_uploaded_file.id = "file123"
        mock_client_instance.files.upload_and_poll.return_value = mock_uploaded_file
        
        # Mock stream events
        from azure.ai.agents.models import MessageDeltaChunk, ThreadRun, AgentStreamEvent
        
        mock_delta_chunk = Mock(spec=MessageDeltaChunk)
        mock_delta_chunk.text = "File processed successfully"
        
        mock_thread_run = Mock(spec=ThreadRun)
        mock_thread_run.status = "completed"
        
        mock_stream_events = [
            (AgentStreamEvent.THREAD_MESSAGE_DELTA, mock_delta_chunk, None),
            (AgentStreamEvent.THREAD_RUN_COMPLETED, mock_thread_run, None)
        ]
        
        mock_stream_context = MagicMock()
        mock_stream_context.__enter__.return_value = mock_stream_events
        mock_stream_context.__exit__.return_value = None
        mock_client_instance.runs.stream.return_value = mock_stream_context
        
        # Mock get_last_message_text_by_role
        mock_response_message = Mock()
        mock_response_message.text.value = "File processed successfully"
        mock_response_message.text.annotations = []
        mock_client_instance.messages.get_last_message_text_by_role.return_value = mock_response_message
        
        # Mock messages.list for image processing
        mock_client_instance.messages.list.return_value = []
        
        # Execute function
        result = await chat_agent("Process this file")
        
        # Verify file upload was called
        mock_client_instance.files.upload_and_poll.assert_called_once()
        upload_call = mock_client_instance.files.upload_and_poll.call_args
        assert upload_call[1]["file_path"] == "/path/to/test.txt"
        
        # Verify message creation with attachment
        create_call = mock_client_instance.messages.create.call_args
        assert len(create_call[1]["attachments"]) == 1
        assert create_call[1]["attachments"][0].file_id == "file123"

    @patch('utils.foundry.cl.user_session')
    @patch('utils.foundry.cl.Message')
    @patch('utils.foundry.get_llm_models')
    @patch('utils.foundry.AgentsClient')
    @patch('utils.foundry.DefaultAzureCredential')
    async def test_chat_agent_with_image_generation(self, mock_credential, mock_agents_client, 
                                                   mock_get_llm_models, mock_message_class, mock_user_session):
        """Test chat agent with image generation."""
        # Mock setup
        mock_user_session.get.side_effect = lambda key, default=None: {
            "chat_settings": self.mock_settings,
            "chat_profile": "foundry/gpt-4.1",
            "thread_id": "thread123",
            "file_uploads": [],
            "file_contents": [],
            "uploaded_files": [],
            "start_time": 1234567880
        }.get(key, default)
        
        mock_message_instance = Mock()  # Use regular Mock instead of AsyncMock
        mock_message_instance.content = ""
        mock_message_instance.elements = []  # Initialize elements as list
        mock_message_instance.send = AsyncMock(return_value=mock_message_instance)
        mock_message_instance.update = AsyncMock()
        mock_message_class.return_value = mock_message_instance
        
        mock_get_llm_models.return_value = [self.mock_llm_details]
        
        # Mock AgentsClient
        mock_client_instance = Mock()
        mock_agents_client.return_value = mock_client_instance
        
        # Mock stream events
        from azure.ai.agents.models import MessageDeltaChunk, ThreadRun, AgentStreamEvent
        
        mock_delta_chunk = Mock(spec=MessageDeltaChunk)
        mock_delta_chunk.text = "I've created an image for you"
        
        mock_thread_run = Mock(spec=ThreadRun)
        mock_thread_run.status = "completed"
        
        mock_stream_events = [
            (AgentStreamEvent.THREAD_MESSAGE_DELTA, mock_delta_chunk, None),
            (AgentStreamEvent.THREAD_RUN_COMPLETED, mock_thread_run, None)
        ]
        
        mock_stream_context = MagicMock()
        mock_stream_context.__enter__.return_value = mock_stream_events
        mock_stream_context.__exit__.return_value = None
        mock_client_instance.runs.stream.return_value = mock_stream_context
        
        # Mock image content in messages
        mock_image_content = Mock()
        mock_image_content.file_id = "img123"
        # Make the mock support the 'in' operator by making it dict-like
        mock_image_content.__contains__ = lambda self, key: key == "file_id"
        
        mock_message_with_image = Mock()
        mock_message_with_image.image_contents = [mock_image_content]
        
        mock_client_instance.messages.list.return_value = [mock_message_with_image]
        
        # Mock get_last_message_text_by_role
        mock_response_message = Mock()
        mock_response_message.text.value = "I've created an image for you"
        mock_response_message.text.annotations = []
        mock_client_instance.messages.get_last_message_text_by_role.return_value = mock_response_message
        
        # Mock Path.cwd()
        with patch('utils.foundry.Path.cwd') as mock_cwd:
            mock_cwd.return_value = Path("/current/dir")
            
            # Mock cl.Image
            with patch('utils.foundry.cl.Image') as mock_image_class:
                mock_image_instance = Mock()
                mock_image_class.return_value = mock_image_instance
                
                # Execute function
                result = await chat_agent("Generate an image")
                
                # Verify image was saved and added to message
                mock_client_instance.files.save.assert_called_once()
                save_call = mock_client_instance.files.save.call_args
                assert save_call[1]["file_id"] == "img123"
                assert "img123_image_file.png" in save_call[1]["file_name"]
                
                # Verify image was created and added to message elements
                mock_image_class.assert_called_once()
                assert mock_message_instance.elements == [mock_image_instance]

    @patch('utils.foundry.cl.user_session')
    @patch('utils.foundry.cl.Message')
    @patch('utils.foundry.get_llm_models')
    @patch('utils.foundry.AgentsClient')
    @patch('utils.foundry.DefaultAzureCredential')
    async def test_chat_agent_with_annotations(self, mock_credential, mock_agents_client, 
                                              mock_get_llm_models, mock_message_class, mock_user_session):
        """Test chat agent with URL annotations."""
        # Mock setup
        mock_user_session.get.side_effect = lambda key, default=None: {
            "chat_settings": self.mock_settings,
            "chat_profile": "foundry/gpt-4.1",
            "thread_id": "thread123",
            "file_uploads": [],
            "file_contents": [],
            "uploaded_files": [],
            "start_time": 1234567880
        }.get(key, default)
        
        mock_message_instance = AsyncMock()
        mock_message_instance.content = ""
        mock_message_class.return_value = mock_message_instance
        
        mock_get_llm_models.return_value = [self.mock_llm_details]
        
        # Mock AgentsClient
        mock_client_instance = Mock()
        mock_agents_client.return_value = mock_client_instance
        
        # Mock stream events
        from azure.ai.agents.models import MessageDeltaChunk, ThreadRun, AgentStreamEvent
        
        mock_delta_chunk = Mock(spec=MessageDeltaChunk)
        mock_delta_chunk.text = "Response with sources"
        
        mock_thread_run = Mock(spec=ThreadRun)
        mock_thread_run.status = "completed"
        
        mock_stream_events = [
            (AgentStreamEvent.THREAD_MESSAGE_DELTA, mock_delta_chunk, None),
            (AgentStreamEvent.THREAD_RUN_COMPLETED, mock_thread_run, None)
        ]
        
        mock_stream_context = MagicMock()
        mock_stream_context.__enter__.return_value = mock_stream_events
        mock_stream_context.__exit__.return_value = None
        mock_client_instance.runs.stream.return_value = mock_stream_context
        
        # Mock annotations
        mock_annotation = Mock()
        mock_annotation.url_citation.title = "Test Source"
        mock_annotation.url_citation.url = "https://example.com/test"
        # Make the mock support the 'in' operator
        mock_annotation.__contains__ = lambda self, key: key == "url_citation"
        
        # Mock get_last_message_text_by_role
        mock_response_message = Mock()
        mock_response_message.text.value = "Response with sources"
        mock_response_message.text.annotations = [mock_annotation]
        mock_client_instance.messages.get_last_message_text_by_role.return_value = mock_response_message
        
        # Mock messages.list for image processing
        mock_client_instance.messages.list.return_value = []
        
        # Execute function
        result = await chat_agent("Question about sources")
        
        # Verify annotations were added to response
        assert "[Test Source](https://example.com/test)" in result

    @patch('utils.foundry.cl.user_session')
    @patch('utils.foundry.cl.Message')
    @patch('utils.foundry.get_llm_models')
    @patch('utils.foundry.AgentsClient')
    @patch('utils.foundry.DefaultAzureCredential')
    async def test_chat_agent_run_failed(self, mock_credential, mock_agents_client, 
                                        mock_get_llm_models, mock_message_class, mock_user_session):
        """Test chat agent when run fails."""
        # Mock setup
        mock_user_session.get.side_effect = lambda key, default=None: {
            "chat_settings": self.mock_settings,
            "chat_profile": "foundry/gpt-4.1",
            "thread_id": "thread123",
            "file_uploads": [],
            "file_contents": [],
            "uploaded_files": [],
            "start_time": 1234567880
        }.get(key, default)
        
        mock_message_instance = AsyncMock()
        mock_message_class.return_value = mock_message_instance
        
        mock_get_llm_models.return_value = [self.mock_llm_details]
        
        # Mock AgentsClient
        mock_client_instance = Mock()
        mock_agents_client.return_value = mock_client_instance
        
        # Mock failed run
        from azure.ai.agents.models import ThreadRun
        
        mock_thread_run = Mock(spec=ThreadRun)
        mock_thread_run.status = "failed"
        mock_thread_run.last_error = "API rate limit exceeded"
        
        mock_stream_events = [
            (None, mock_thread_run, None)
        ]
        
        mock_stream_context = MagicMock()
        mock_stream_context.__enter__.return_value = mock_stream_events
        mock_stream_context.__exit__.return_value = None
        mock_client_instance.runs.stream.return_value = mock_stream_context
        
        # Execute function and expect RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await chat_agent("Test message")
        
        assert "Error generating response in chat_agent" in str(exc_info.value)

    @patch('utils.foundry.cl.user_session')
    @patch('utils.foundry.cl.Message')
    @patch('utils.foundry.get_llm_models')
    @patch('utils.foundry.AgentsClient')
    @patch('utils.foundry.DefaultAzureCredential')
    async def test_chat_agent_stream_error(self, mock_credential, mock_agents_client, 
                                          mock_get_llm_models, mock_message_class, mock_user_session):
        """Test chat agent when stream error occurs."""
        # Mock setup
        mock_user_session.get.side_effect = lambda key, default=None: {
            "chat_settings": self.mock_settings,
            "chat_profile": "foundry/gpt-4.1",
            "thread_id": "thread123",
            "file_uploads": [],
            "file_contents": [],
            "uploaded_files": [],
            "start_time": 1234567880
        }.get(key, default)
        
        mock_message_instance = AsyncMock()
        mock_message_class.return_value = mock_message_instance
        
        mock_get_llm_models.return_value = [self.mock_llm_details]
        
        # Mock AgentsClient
        mock_client_instance = Mock()
        mock_agents_client.return_value = mock_client_instance
        
        # Mock stream error
        from azure.ai.agents.models import AgentStreamEvent
        
        mock_stream_events = [
            (AgentStreamEvent.ERROR, "Stream error occurred", None)
        ]
        
        mock_stream_context = MagicMock()
        mock_stream_context.__enter__.return_value = mock_stream_events
        mock_stream_context.__exit__.return_value = None
        mock_client_instance.runs.stream.return_value = mock_stream_context
        
        # Execute function and expect RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await chat_agent("Test message")
        
        assert "Error generating response in chat_agent" in str(exc_info.value)

    @patch('utils.foundry.cl.user_session')
    @patch('utils.foundry.cl.Message')
    @patch('utils.foundry.get_llm_models')
    @patch('utils.foundry.AgentsClient')
    @patch('utils.foundry.DefaultAzureCredential')
    async def test_chat_agent_no_response_message(self, mock_credential, mock_agents_client, 
                                                 mock_get_llm_models, mock_message_class, mock_user_session):
        """Test chat agent when no response message is returned."""
        # Mock setup
        mock_user_session.get.side_effect = lambda key, default=None: {
            "chat_settings": self.mock_settings,
            "chat_profile": "foundry/gpt-4.1",
            "thread_id": "thread123",
            "file_uploads": [],
            "file_contents": [],
            "uploaded_files": [],
            "start_time": 1234567880
        }.get(key, default)
        
        mock_message_instance = AsyncMock()
        mock_message_class.return_value = mock_message_instance
        
        mock_get_llm_models.return_value = [self.mock_llm_details]
        
        # Mock AgentsClient
        mock_client_instance = Mock()
        mock_agents_client.return_value = mock_client_instance
        
        # Mock stream events
        from azure.ai.agents.models import ThreadRun, AgentStreamEvent
        
        mock_thread_run = Mock(spec=ThreadRun)
        mock_thread_run.status = "completed"
        
        mock_stream_events = [
            (AgentStreamEvent.THREAD_RUN_COMPLETED, mock_thread_run, None)
        ]
        
        mock_stream_context = MagicMock()
        mock_stream_context.__enter__.return_value = mock_stream_events
        mock_stream_context.__exit__.return_value = None
        mock_client_instance.runs.stream.return_value = mock_stream_context
        
        # Mock get_last_message_text_by_role to return None
        mock_client_instance.messages.get_last_message_text_by_role.return_value = None
        
        # Mock messages.list for image processing
        mock_client_instance.messages.list.return_value = []
        
        # Execute function and expect RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await chat_agent("Test message")
        
        assert "Error generating response in chat_agent" in str(exc_info.value)

    @patch('utils.foundry.cl.user_session')
    @patch('utils.foundry.cl.Message')
    @patch('utils.foundry.get_llm_models')
    async def test_chat_agent_message_creation_failure(self, mock_get_llm_models, mock_message_class, mock_user_session):
        """Test chat agent when message creation fails."""
        # Mock setup
        mock_user_session.get.side_effect = lambda key, default=None: {
            "chat_settings": self.mock_settings,
            "chat_profile": "foundry/gpt-4.1",
            "thread_id": "thread123",
            "file_uploads": [],
            "file_contents": [],
            "uploaded_files": [],
            "start_time": 1234567880
        }.get(key, default)
        
        # Mock message creation to return None (failure)
        mock_message_class.return_value = None
        
        mock_get_llm_models.return_value = [self.mock_llm_details]
        
        # Execute function and expect RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await chat_agent("Test message")
        
        assert "Error generating response in chat_agent" in str(exc_info.value)
        assert "'NoneType' object has no attribute 'send'" in str(exc_info.value)

    @patch('utils.foundry.cl.user_session')
    @patch('utils.foundry.cl.Message')
    @patch('utils.foundry.get_llm_models')
    @patch('utils.foundry.AgentsClient')
    @patch('utils.foundry.DefaultAzureCredential')
    async def test_chat_agent_multiple_files(self, mock_credential, mock_agents_client, 
                                           mock_get_llm_models, mock_message_class, mock_user_session):
        """Test chat agent with multiple file uploads."""
        # Mock setup
        mock_user_session.get.side_effect = lambda key, default=None: {
            "chat_settings": self.mock_settings,
            "chat_profile": "foundry/gpt-4.1",
            "thread_id": "thread123",
            "file_uploads": [
                {"name": "file1.txt", "mime": "text/plain", "path": "/path/to/file1.txt", "base64": None},
                {"name": "file2.pdf", "mime": "application/pdf", "path": "/path/to/file2.pdf", "base64": None}
            ],
            "file_contents": [],
            "uploaded_files": ["/path/to/file1.txt", "/path/to/file2.pdf"],
            "start_time": 1234567880
        }.get(key, default)
        
        mock_message_instance = AsyncMock()
        mock_message_instance.content = ""
        mock_message_class.return_value = mock_message_instance
        
        mock_get_llm_models.return_value = [self.mock_llm_details]
        
        # Mock AgentsClient
        mock_client_instance = Mock()
        mock_agents_client.return_value = mock_client_instance
        
        # Mock file uploads
        mock_uploaded_file1 = Mock()
        mock_uploaded_file1.id = "file123"
        mock_uploaded_file2 = Mock()
        mock_uploaded_file2.id = "file456"
        
        mock_client_instance.files.upload_and_poll.side_effect = [mock_uploaded_file1, mock_uploaded_file2]
        
        # Mock stream events
        from azure.ai.agents.models import MessageDeltaChunk, ThreadRun, AgentStreamEvent
        
        mock_delta_chunk = Mock(spec=MessageDeltaChunk)
        mock_delta_chunk.text = "Files processed"
        
        mock_thread_run = Mock(spec=ThreadRun)
        mock_thread_run.status = "completed"
        
        mock_stream_events = [
            (AgentStreamEvent.THREAD_MESSAGE_DELTA, mock_delta_chunk, None),
            (AgentStreamEvent.THREAD_RUN_COMPLETED, mock_thread_run, None)
        ]
        
        mock_stream_context = MagicMock()
        mock_stream_context.__enter__.return_value = mock_stream_events
        mock_stream_context.__exit__.return_value = None
        mock_client_instance.runs.stream.return_value = mock_stream_context
        
        # Mock get_last_message_text_by_role
        mock_response_message = Mock()
        mock_response_message.text.value = "Files processed"
        mock_response_message.text.annotations = []
        mock_client_instance.messages.get_last_message_text_by_role.return_value = mock_response_message
        
        # Mock messages.list for image processing
        mock_client_instance.messages.list.return_value = []
        
        # Execute function
        result = await chat_agent("Process these files")
        
        # Verify both files were uploaded
        assert mock_client_instance.files.upload_and_poll.call_count == 2
        
        # Verify message creation with multiple attachments
        create_call = mock_client_instance.messages.create.call_args
        assert len(create_call[1]["attachments"]) == 2
        assert create_call[1]["attachments"][0].file_id == "file123"
        assert create_call[1]["attachments"][1].file_id == "file456"


if __name__ == "__main__":
    pytest.main([__file__])
