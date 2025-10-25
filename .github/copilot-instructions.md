# Chainlit GPT - BSP AI Assistant

## Architecture Overview

This is a **Chainlit-based conversational AI** for Bangko Sentral ng Pilipinas (BSP) employees with dual LLM provider support:

- **Primary Route**: `utils/chats.py` → LiteLLM → Multiple providers (Azure OpenAI, Gemini, Perplexity) 
- **Azure AI Foundry Route**: `utils/foundry.py` → Azure AI Agents SDK → Agent-based interactions with file processing

## Key Components & Data Flow

```
app.py (Chainlit handlers) → utils/utils.py (core utilities) → {chats.py | foundry.py} → LLM Providers
```

- **`app.py`**: Chainlit event handlers (`@cl.on_message`, `@cl.on_chat_start`, chat profiles, auth)
- **`utils/utils.py`**: Session management, logging, message formatting, model configuration
- **`utils/chats.py`**: LiteLLM-based completion for standard providers 
- **`utils/foundry.py`**: Azure AI Agents for advanced file processing + code interpretation

## Code standards

### Required Before Each Commit

- Run Python tests to ensure backend functionality
- When adding new functionality, make sure you update the README documentation
- Make sure all guidance in the Copilot Instructions file is updated with any relevant changes, including to project structure and scripts, and programming guidance

### Code formatting requirements

- When writing Python, you must use type hints for return values and function parameters.
- Every function should have docstrings or the language equivalent
- Before imports or any code, add a comment block that explains the purpose of the file.

### Styling

- Maintain dark mode theme throughout the application
- Use rounded corners for UI elements
- Follow modern UI/UX principles with clean, accessible interfaces

### GitHub Actions workflows

- Follow good security practices
- Make sure to explicitly set the workflow permissions
- Add comments to document what tasks are being performed

## Critical Configuration Patterns

### Model Configuration (`llm_config/*.json`)
Models are defined with provider-specific routing:
```json
{
  "model_deployment": "foundry/gpt-4.1",  // Routes to foundry.py
  "model_deployment": "perplexity/sonar", // Routes to chats.py via LiteLLM
  "api_endpoint": "https://...",          // Required for Azure AI Foundry
  "model_id": "asst_...",                 // Required for Azure AI Agents
}
```

### Provider Routing Logic
- **Foundry models**: `model_provider == "foundry"` → `chat_agent()` in `foundry.py`
- **All others**: `chat_completion()` in `chats.py` via LiteLLM

### Session Context & Logging
Enhanced logging automatically captures session + user context:
```python
# Logs automatically include: [session_id] [user_id] message
logger.info("User action")  # → [sess_123] [alice] ->>>>> User action
```

## Critical Workflows

### Development Setup
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
chainlit run app.py --watch
```

### Model Configuration Workflow
1. Add model to `llm_config/llm_config.json` 
2. Set `LLM_CONFIG` env var to JSON array (for production) OR use file (dev)
3. Model appears in chat profiles automatically via `@cl.set_chat_profiles`

### File Upload Handling
- **Foundry route**: Files uploaded to Azure AI Agents with `CodeInterpreterTool`
- **Standard route**: Files converted to base64 (images) or markdown text via `MarkItDown`

## Project-Specific Conventions

### Error Handling Pattern
```python
try:
    # operation
    logger.info("Success message")
except Exception as e:
    await cl.Message(content=f"An error occurred: {str(e)}", author="Error").send()
    logger.error(f"Error: {str(e)}")
```

### Message Structure Convention
OpenAI format with BSP-specific system prompt in `init_settings()`:
```python
[
  {"role": "system", "content": [{"type": "text", "text": "BSP AI Assistant instructions..."}]},
  {"role": "user", "content": [{"type": "text", "text": "..."}]}
]
```

### Chat History Management
- **Auto-pruning**: Keeps last 10 messages when assistant responds
- **Session storage**: `cl.user_session.set("chat_history", chat_history)`
- **System prompt**: Always prepended from current settings

## Authentication & Authorization

Uses **header-based auth** for Azure App Service integration:
- `X-MS-CLIENT-PRINCIPAL-NAME` → user identifier
- `X-MS-CLIENT-PRINCIPAL-ID` → user ID for logging context
- Default fallback: `dummy@microsoft.com` / `9876543210`

## Environment Variables

**Required**:
- `LLM_CONFIG`: JSON array of model configurations (production)
- `CHAINLIT_AUTH_SECRET`: Required for Chainlit auth

**Provider-specific** (based on models in config):
- `AZURE_OPENAI_*`: For Azure OpenAI models  
- `AIPROJECT_CONNECTION_STRING`: For Azure AI Foundry models

## Testing & Debugging

### Logging System
- **Console**: Color-coded with session/user context
- **File**: `logs/app.log` (10MB rotation, 30-day retention)

### Model Testing
Switch between providers via Chainlit chat profiles (no code changes needed).

### Unit Testing
- Run all tests by executing `python run_tests.py unit`
- Run with coverage by executing `python run_tests.py coverage`
- Once all tests pass, ensure coverage is at least 90% and document any gaps in `docs/test-results.md`

## Integration Points

- **Azure AI Foundry**: Agent-based interactions with file uploads
- **LiteLLM**: Unified interface for multiple LLM providers
- **MarkItDown**: File-to-text conversion for non-foundry models
- **Chainlit**: UI framework with built-in auth, file uploads, streaming

## Common Patterns

When adding new providers, update `llm_config.json` and ensure `get_llm_params()` handles provider-specific parameters. Foundry models require `model_id` and `api_endpoint`; others use LiteLLM conventions.
