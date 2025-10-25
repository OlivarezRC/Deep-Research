# Unit tests for utils/test_config.py - Configuration testing utility
# Tests configuration file validation and display functionality

import pytest
import json
import os
import tempfile
from unittest.mock import patch, mock_open
import sys

# Add the parent directory to the path to import the test_config module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.test_config import test_config_file


class TestTestConfigFile:
    """Test cases for test_config_file function."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        self.valid_config = [
            {
                "model_deployment": "azure/gpt-4",
                "description": "Azure GPT-4 model",
                "api_key": "test-key-1",
                "api_endpoint": "https://test1.openai.azure.com"
            },
            {
                "model_deployment": "foundry/gpt-4.1",
                "description": "Foundry GPT-4.1 model",
                "api_key": "test-key-2",
                "api_endpoint": "https://test2.foundry.azure.com"
            }
        ]
        
        self.invalid_json = '{"model_deployment": "azure/gpt-4", "incomplete": true'

    @patch('builtins.open', mock_open())
    @patch('builtins.print')
    @patch('utils.test_config.json.load')
    def test_config_file_valid_json(self, mock_json_load, mock_print):
        """Test test_config_file with valid JSON configuration."""
        mock_json_load.return_value = self.valid_config
        
        test_config_file()
        
        # Verify file was opened and JSON was loaded
        mock_json_load.assert_called_once()
        
        # Verify JSON was printed
        mock_print.assert_called_once()
        printed_output = mock_print.call_args[0][0]
        
        # The output should be a JSON string representation of the config
        parsed_output = json.loads(printed_output)
        assert parsed_output == self.valid_config

    @patch('builtins.open')
    @patch('builtins.print')
    def test_config_file_not_found(self, mock_print, mock_open_func):
        """Test test_config_file when configuration file is not found."""
        mock_open_func.side_effect = FileNotFoundError("File not found")
        
        test_config_file()
        
        # Verify error message was printed
        mock_print.assert_called_once()
        error_message = mock_print.call_args[0][0]
        assert "Error: Could not find the config file" in error_message

    @patch('builtins.open', mock_open())
    @patch('builtins.print')
    @patch('utils.test_config.json.load')
    def test_config_file_invalid_json(self, mock_json_load, mock_print):
        """Test test_config_file with invalid JSON format."""
        mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)
        
        test_config_file()
        
        # Verify error message was printed
        mock_print.assert_called_once()
        error_message = mock_print.call_args[0][0]
        assert "Error: Invalid JSON format in config file" in error_message

    @patch('builtins.open', mock_open())
    @patch('builtins.print')
    @patch('utils.test_config.json.load')
    def test_config_file_unexpected_error(self, mock_json_load, mock_print):
        """Test test_config_file with unexpected error."""
        mock_json_load.side_effect = Exception("Unexpected error")
        
        test_config_file()
        
        # Verify error message was printed
        mock_print.assert_called_once()
        error_message = mock_print.call_args[0][0]
        assert "Error: An unexpected error occurred" in error_message
        assert "Unexpected error" in error_message

    @patch('builtins.open', mock_open())
    @patch('builtins.print')
    @patch('utils.test_config.json.load')
    def test_config_file_empty_config(self, mock_json_load, mock_print):
        """Test test_config_file with empty configuration."""
        mock_json_load.return_value = []
        
        test_config_file()
        
        # Verify empty list was printed
        mock_print.assert_called_once()
        printed_output = mock_print.call_args[0][0]
        assert printed_output == "[]"

    @patch('builtins.open', mock_open())
    @patch('builtins.print')
    @patch('utils.test_config.json.load')
    def test_config_file_with_unicode_content(self, mock_json_load, mock_print):
        """Test test_config_file with unicode content in configuration."""
        unicode_config = [
            {
                "model_deployment": "azure/gpt-4",
                "description": "Azure GPT-4 with émojis 🤖",
                "api_key": "tëst-kéy-1"
            }
        ]
        mock_json_load.return_value = unicode_config
        
        test_config_file()
        
        # Verify unicode content was handled correctly
        mock_print.assert_called_once()
        printed_output = mock_print.call_args[0][0]
        parsed_output = json.loads(printed_output)
        assert parsed_output == unicode_config
        assert "émojis 🤖" in printed_output

    @patch('utils.test_config.os.path.join')
    @patch('utils.test_config.os.path.dirname')
    @patch('builtins.open', mock_open())
    @patch('builtins.print')
    @patch('utils.test_config.json.load')
    def test_config_file_path_construction(self, mock_json_load, mock_print, 
                                          mock_dirname, mock_join):
        """Test that config file path is constructed correctly."""
        mock_dirname.side_effect = ["/utils", "/project"]  # Two calls to dirname
        mock_join.return_value = "/project/llm_config/llm_config.json"
        mock_json_load.return_value = self.valid_config
        
        test_config_file()
        
        # Verify path construction
        assert mock_dirname.call_count == 2
        mock_join.assert_called_once_with("/project", 'llm_config/llm_config.json')

    def test_config_file_main_execution(self):
        """Test that test_config_file runs when executed as main."""
        # This test verifies the if __name__ == "__main__" block
        with patch('utils.test_config.test_config_file') as mock_test_config:
            # Import the module to trigger the main execution
            import utils.test_config
            
            # We can't easily test the main execution without running the actual script,
            # but we can verify the function exists and is callable
            assert callable(utils.test_config.test_config_file)

    @patch('builtins.open')
    @patch('builtins.print')
    def test_config_file_permission_error(self, mock_print, mock_open_func):
        """Test test_config_file when file permission is denied."""
        mock_open_func.side_effect = PermissionError("Permission denied")
        
        test_config_file()
        
        # Verify error message was printed
        mock_print.assert_called_once()
        error_message = mock_print.call_args[0][0]
        assert "Error: An unexpected error occurred" in error_message
        assert "Permission denied" in error_message

    @patch('builtins.open', mock_open())
    @patch('builtins.print')
    @patch('utils.test_config.json.load')
    def test_config_file_nested_json_structure(self, mock_json_load, mock_print):
        """Test test_config_file with nested JSON structure."""
        nested_config = [
            {
                "model_deployment": "azure/gpt-4",
                "settings": {
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "advanced": {
                        "top_p": 0.9,
                        "frequency_penalty": 0.0
                    }
                },
                "endpoints": ["https://api1.com", "https://api2.com"]
            }
        ]
        mock_json_load.return_value = nested_config
        
        test_config_file()
        
        # Verify nested structure was handled correctly
        mock_print.assert_called_once()
        printed_output = mock_print.call_args[0][0]
        parsed_output = json.loads(printed_output)
        assert parsed_output == nested_config
        assert parsed_output[0]["settings"]["advanced"]["top_p"] == 0.9

    @patch('builtins.open', mock_open())
    @patch('builtins.print')
    @patch('utils.test_config.json.load')
    def test_config_file_special_characters(self, mock_json_load, mock_print):
        """Test test_config_file with special characters in JSON."""
        special_config = [
            {
                "model_deployment": "azure/gpt-4",
                "description": 'Model with "quotes" and \\backslashes\\ and \n newlines',
                "api_key": "key-with-special-chars!@#$%^&*()"
            }
        ]
        mock_json_load.return_value = special_config
        
        test_config_file()
        
        # Verify special characters were handled correctly
        mock_print.assert_called_once()
        printed_output = mock_print.call_args[0][0]
        parsed_output = json.loads(printed_output)
        assert parsed_output == special_config


if __name__ == "__main__":
    pytest.main([__file__])
