# Test Specifications - BSP AI Assistant

## Detailed Test Case Documentation

This document provides detailed specifications for each test case in the BSP AI Assistant test suite.

## Test File: `test_app.py`

### TestHeaderAuthCallback Class

#### `test_header_auth_callback_with_headers()`
- **Purpose:** Verify authentication with valid Azure App Service headers
- **Mocked Headers:** `X-MS-CLIENT-PRINCIPAL-NAME`, `X-MS-CLIENT-PRINCIPAL-ID`
- **Expected Result:** User object with correct identifier and metadata
- **Assertions:** User ID, identifier, and display name validation

#### `test_header_auth_callback_without_headers()`
- **Purpose:** Test fallback authentication when headers are missing
- **Expected Result:** Default user object (`dummy@microsoft.com`)
- **Assertions:** Fallback user credentials are properly assigned

### TestChatProfile Class

#### `test_set_chat_profiles_multiple_models()`
- **Purpose:** Verify chat profile creation with multiple LLM models
- **Test Data:** Mock model configurations for different providers
- **Expected Result:** Chat profiles created for each model
- **Assertions:** Profile names, icons, and markdown content

#### `test_set_chat_profiles_single_model()`
- **Purpose:** Test profile creation with single model configuration
- **Expected Result:** One chat profile with correct metadata
- **Assertions:** Single profile properties and availability

#### `test_set_chat_profiles_empty_models()`
- **Purpose:** Handle edge case of no available models
- **Expected Result:** No profiles created, graceful handling
- **Assertions:** Empty profile list, no errors raised

### TestStart Class

#### `test_on_chat_start_with_settings()`
- **Purpose:** Test chat initialization with valid settings
- **Mocked Components:** Session settings, message sending
- **Expected Result:** Welcome message sent, settings initialized
- **Assertions:** Message content, settings storage, logger calls

#### `test_on_chat_start_without_settings()`
- **Purpose:** Handle initialization when settings are unavailable
- **Expected Result:** Default behavior, error handling
- **Assertions:** Graceful degradation, appropriate logging

### TestMain Class

#### `test_on_message_standard_provider()`
- **Purpose:** Test message routing to standard LLM providers
- **Test Scenario:** Non-foundry model selection
- **Expected Result:** Message routed to `chat_completion()`
- **Assertions:** Function call verification, parameter passing

#### `test_on_message_foundry_provider()`
- **Purpose:** Test message routing to Azure AI Foundry
- **Test Scenario:** Foundry model selection
- **Expected Result:** Message routed to `chat_agent()`
- **Assertions:** Correct routing logic, agent invocation

## Test File: `test_utils.py`

### TestTruncate Class

#### `test_truncate_no_truncation_needed()`
- **Purpose:** Verify text shorter than limit passes through unchanged
- **Test Data:** Short text with various word counts
- **Expected Result:** Original text returned
- **Assertions:** Text integrity, no modification

#### `test_truncate_at_word_boundary()`
- **Purpose:** Test intelligent word boundary truncation
- **Test Data:** Text requiring truncation at word boundaries
- **Expected Result:** Clean truncation without partial words
- **Assertions:** Word boundary respect, proper length

#### `test_truncate_single_long_word()`
- **Purpose:** Handle edge case of single word exceeding limit
- **Test Data:** Individual word longer than character limit
- **Expected Result:** Hard truncation with ellipsis
- **Assertions:** Character limit enforcement, ellipsis addition

### TestAddContext Class

#### `test_add_context_new_key()`
- **Purpose:** Test adding new key-value pair to session context
- **Mocked Components:** Chainlit user session
- **Expected Result:** Key-value pair stored in session
- **Assertions:** Session storage call, correct parameters

#### `test_add_context_existing_key()`
- **Purpose:** Test updating existing context key
- **Expected Result:** Value updated in session storage
- **Assertions:** Overwrite behavior, correct value storage

#### `test_add_context_error_handling()`
- **Purpose:** Test error handling when session operations fail
- **Test Scenario:** Mocked session exception
- **Expected Result:** Error logged, function continues
- **Assertions:** Exception handling, logging behavior

### TestGetLogger Class

#### `test_get_logger_with_session_context()`
- **Purpose:** Test logger creation with session and user context
- **Test Data:** Mocked session ID and user information
- **Expected Result:** Logger with contextual formatting
- **Assertions:** Logger configuration, context injection

#### `test_get_logger_without_session()`
- **Purpose:** Test logger creation when session unavailable
- **Expected Result:** Basic logger without session context
- **Assertions:** Fallback logger creation, default configuration

### TestGetLlmModels Class

#### `test_get_llm_models_from_env()`
- **Purpose:** Test model loading from environment variable
- **Test Data:** JSON string in `LLM_CONFIG` environment variable
- **Expected Result:** Parsed model configurations
- **Assertions:** Environment variable parsing, model structure

#### `test_get_llm_models_from_file()`
- **Purpose:** Test fallback to file-based configuration
- **Test Scenario:** No environment variable set
- **Expected Result:** Models loaded from JSON file
- **Assertions:** File reading, JSON parsing, model validation

#### `test_get_llm_models_from_env_invalid_json()`
- **Purpose:** Test error handling for malformed environment JSON
- **Test Data:** Invalid JSON string in environment variable
- **Expected Result:** Graceful fallback to file reading
- **Assertions:** Error handling, fallback mechanism

#### `test_get_llm_models_from_file_when_no_env()`
- **Purpose:** Test file loading when environment variable is empty
- **Test Scenario:** Empty or missing `LLM_CONFIG`
- **Expected Result:** File-based configuration loading
- **Assertions:** Empty environment detection, file fallback

### TestAppendMessage Class

#### `test_append_message_new_chat_history()`
- **Purpose:** Test message appending to empty chat history
- **Test Data:** New message object, empty history
- **Expected Result:** Message added to history, auto-pruning disabled
- **Assertions:** History initialization, message addition

#### `test_append_message_existing_chat_history()`
- **Purpose:** Test message appending to existing conversation
- **Test Data:** Existing messages, new message to append
- **Expected Result:** Message added, history maintained
- **Assertions:** History preservation, proper appending

#### `test_append_message_auto_prune_enabled()`
- **Purpose:** Test automatic history pruning for long conversations
- **Test Data:** Long conversation history, auto-prune enabled
- **Expected Result:** History trimmed to last 10 messages
- **Assertions:** Pruning logic, message count validation

#### `test_append_message_system_prompt_preservation()`
- **Purpose:** Test system prompt preservation during pruning
- **Test Data:** History with system prompt and many messages
- **Expected Result:** System prompt retained, user messages pruned
- **Assertions:** System message preservation, selective pruning

### TestInitSettings Class

#### `test_init_settings_default_configuration()`
- **Purpose:** Test settings initialization with default values
- **Expected Result:** BSP AI Assistant configuration loaded
- **Assertions:** System prompt content, default parameters

#### `test_init_settings_custom_model_provider()`
- **Purpose:** Test settings with specific model provider
- **Test Data:** Custom provider configuration
- **Expected Result:** Provider-specific settings applied
- **Assertions:** Provider configuration, model parameters

## Test File: `test_chats.py`

### TestGetLlmParams Class

#### `test_get_llm_params_azure_provider()`
- **Purpose:** Test parameter extraction for Azure OpenAI models
- **Test Data:** Azure model configuration
- **Expected Result:** Azure-specific parameters extracted
- **Assertions:** API endpoint, deployment name, API version

#### `test_get_llm_params_non_azure_provider()`
- **Purpose:** Test parameter extraction for non-Azure providers
- **Test Data:** Generic LLM provider configuration
- **Expected Result:** Standard LLM parameters
- **Assertions:** Model name, provider settings

#### `test_get_llm_params_missing_keys()`
- **Purpose:** Test error handling for incomplete configurations
- **Test Data:** Configuration missing required keys
- **Expected Result:** Default values assigned, no errors
- **Assertions:** Graceful degradation, default parameters

### TestChatCompletion Class

#### `test_chat_completion_success()`
- **Purpose:** Test successful LLM completion request
- **Mocked Components:** LiteLLM completion response
- **Expected Result:** Formatted response with citations
- **Assertions:** Response processing, citation extraction

#### `test_chat_completion_with_streaming()`
- **Purpose:** Test streaming response handling
- **Test Data:** Chunked response simulation
- **Expected Result:** Progressive message updates
- **Assertions:** Chunk processing, real-time updates

#### `test_chat_completion_with_citations()`
- **Purpose:** Test citation processing and display
- **Test Data:** Response with source citations
- **Expected Result:** Citations formatted and displayed
- **Assertions:** Citation parsing, source attribution

#### `test_chat_completion_error_handling()`
- **Purpose:** Test error handling for LLM failures
- **Test Scenario:** LLM service unavailable
- **Expected Result:** Error message sent to user
- **Assertions:** Error detection, user notification

## Test File: `test_foundry.py`

### TestChatAgent Class

#### `test_chat_agent_success()`
- **Purpose:** Test successful Azure AI agent interaction
- **Mocked Components:** Azure AI client, agent response
- **Expected Result:** Agent response processed and displayed
- **Assertions:** Agent invocation, response handling

#### `test_chat_agent_with_file_upload()`
- **Purpose:** Test file upload processing with agent tools
- **Test Data:** File upload simulation
- **Expected Result:** File processed by code interpreter
- **Assertions:** File handling, tool integration

#### `test_chat_agent_streaming_events()`
- **Purpose:** Test real-time event streaming from agent
- **Test Data:** Simulated streaming events
- **Expected Result:** Progressive message updates
- **Assertions:** Event processing, stream handling

#### `test_chat_agent_image_generation()`
- **Purpose:** Test image generation through agent tools
- **Test Scenario:** Image creation request
- **Expected Result:** Image displayed in chat
- **Assertions:** Image processing, display functionality

## Test File: `test_test_config.py`

### TestTestConfigFile Class

#### `test_test_config_file_valid_json()`
- **Purpose:** Test configuration validation with valid JSON
- **Test Data:** Well-formed JSON configuration
- **Expected Result:** Validation success, no errors
- **Assertions:** JSON parsing, validation logic

#### `test_test_config_file_invalid_json()`
- **Purpose:** Test error handling for malformed JSON
- **Test Data:** Syntactically incorrect JSON
- **Expected Result:** Validation failure, error reported
- **Assertions:** Error detection, failure reporting

#### `test_test_config_file_unicode_support()`
- **Purpose:** Test Unicode character handling in configuration
- **Test Data:** Configuration with international characters
- **Expected Result:** Proper Unicode processing
- **Assertions:** Unicode support, character encoding

#### `test_test_config_file_missing_file()`
- **Purpose:** Test handling of missing configuration files
- **Test Scenario:** File not found error
- **Expected Result:** Appropriate error handling
- **Assertions:** File error detection, error messages

## Mock Strategy Details

### Chainlit Framework Mocking
```python
# Session management
@patch('chainlit.user_session.get')
@patch('chainlit.user_session.set')

# Message handling
@patch('chainlit.Message')
message_mock.send = AsyncMock()

# Authentication
@patch('chainlit.header_auth_callback')
```

### Azure AI Services Mocking
```python
# Client creation
@patch('azure.ai.projects.aio.AIProjectClient')

# Agent interactions
mock_agent.return_value.messages.create_and_run_stream.return_value = mock_stream
```

### LiteLLM Provider Mocking
```python
# Completion requests
@patch('litellm.acompletion')
mock_completion.return_value = mock_response

# Streaming responses
mock_response.choices[0].delta.content = "chunk"
```

### File System Mocking
```python
# File operations
@patch('builtins.open', new_callable=mock_open)
mock_file.read_data = '{"key": "value"}'

# JSON parsing
@patch('json.load')
mock_json.return_value = test_data
```

## Performance Testing Notes

### Execution Time Benchmarks
- **Individual Test Average:** 0.1-0.5 seconds
- **Test Class Average:** 2-8 seconds
- **Full Suite Runtime:** 12-40 seconds (depends on system)

### Memory Usage Patterns
- **Mock Object Overhead:** Minimal with proper cleanup
- **File Operation Simulation:** Memory-efficient with mock_open
- **Async Test Handling:** Optimized with pytest-asyncio

### CI/CD Optimization
- **Parallel Execution:** Safe for independent tests
- **Resource Cleanup:** Automatic with pytest fixtures
- **Caching Strategy:** Dependencies cached between runs
