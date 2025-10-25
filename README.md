# Chainlit GPT - BSP AI Assistant

A sophisticated conversational AI application built for Bangko Sentral ng Pilipinas (BSP) employees, providing dual LLM provider support with advanced file processing capabilities.

## 🏛️ About BSP AI Assistant

The BSP AI Assistant is designed specifically for internal use by Bangko Sentral ng Pilipinas employees. It serves as an intelligent productivity tool that helps with information retrieval, task assistance, problem-solving, and workflow optimization while maintaining strict confidentiality and compliance standards.

## 🏗️ Architecture Overview

The application features a dual-route architecture supporting multiple LLM providers:

- **Primary Route**: `utils/chats.py` → LiteLLM → Multiple providers (Azure OpenAI, Gemini, Perplexity)
- **Azure AI Foundry Route**: `utils/foundry.py` → Azure AI Agents SDK → Agent-based interactions with advanced file processing

```
app.py (Chainlit handlers) → utils/utils.py (core utilities) → {chats.py | foundry.py} → LLM Providers
```

## ✨ Key Features

### 🤖 Multi-Provider LLM Support
- **Azure OpenAI**: GPT-4, GPT-4o, and other Azure-hosted models
- **Azure AI Foundry**: Agent-based interactions with code interpretation
- **Google Gemini**: Advanced reasoning and multimodal capabilities
- **Perplexity**: Real-time web search and up-to-date information
- **Azure Model Catalog**: Phi-4 and other cutting-edge models

### 📁 Advanced File Processing
- **Image Analysis**: Base64 encoding for visual content processing
- **Document Conversion**: MarkItDown integration for multiple file formats
- **Azure AI Foundry**: Advanced file processing with CodeInterpreter
- **Multi-format Support**: PDF, DOCX, images, spreadsheets, and more

### 🔐 Enterprise Authentication
- **Azure App Service Integration**: Header-based authentication
- **Role-based Access**: Configurable user roles and permissions
- **Session Management**: Secure session handling with user context

### 💬 Intelligent Chat Features
- **Chat Profiles**: Dynamic model selection interface
- **Conversation Starters**: Pre-configured helpful prompts
- **Chat History**: Automatic pruning with 10-message retention
- **Streaming Responses**: Real-time response generation
- **File Attachments**: Drag-and-drop file upload support

### 📊 Comprehensive Logging
- **Enhanced Logging**: Session and user context tracking
- **File Rotation**: 10MB log files with 30-day retention
- **Color-coded Console**: Development-friendly terminal output
- **Error Tracking**: Detailed error logging and debugging

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Azure subscription (for Azure AI services)
- Required API keys for chosen LLM providers

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd chainlitgpt
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp sample.env .env
   # Edit .env with your API keys and configuration
   ```

5. **Set up LLM configuration**
   ```bash
   cp sample_llm_config.json llm_config/llm_config.json
   # Edit with your model configurations
   ```

6. **Run the application**
   ```bash
   chainlit run app.py --watch
   ```

## ⚙️ Configuration

### Model Configuration

Models are configured in `llm_config/llm_config.json` or via the `LLM_CONFIG` environment variable:

```json
[
    {
        "model_id": "asst_xyz123",
        "model_deployment": "foundry/gpt-4o",
        "api_key": "your-api-key",
        "api_version": null,
        "api_endpoint": "https://your-foundry-endpoint",
        "description": "Azure AI Foundry agent with code interpretation"
    },
    {
        "model_id": "gpt-4o",
        "model_deployment": "azure/gpt-4o",
        "api_key": "your-azure-key",
        "api_version": "2024-05-01-preview",
        "api_endpoint": "https://your-resource.openai.azure.com",
        "description": "Azure OpenAI GPT-4o model"
    }
]
```

### Environment Variables

**Required:**
- `LLM_CONFIG`: JSON array of model configurations (production)
- `CHAINLIT_AUTH_SECRET`: Required for Chainlit authentication

**Provider-specific:**
- `AZURE_OPENAI_*`: For Azure OpenAI models
- `AIPROJECT_CONNECTION_STRING`: For Azure AI Foundry models
- `GOOGLE_API_KEY`: For Gemini models
- `PERPLEXITY_API_KEY`: For Perplexity models

### Provider Routing

The application automatically routes requests based on model configuration:
- **Foundry models**: `model_provider == "foundry"` → `chat_agent()` in `foundry.py`
- **All others**: `chat_completion()` in `chats.py` via LiteLLM

## 🔧 Development

### Project Structure

```
├── app.py                    # Main Chainlit application
├── utils/
│   ├── utils.py             # Core utilities and session management
│   ├── chats.py             # LiteLLM-based chat completion
│   └── foundry.py           # Azure AI Foundry agent integration
├── llm_config/              # Model configuration files
├── public/                  # Static assets and custom styling
├── logs/                    # Application logs
└── tests/                   # Test files
```

### Key Components

- **`app.py`**: Chainlit event handlers, authentication, chat profiles
- **`utils/utils.py`**: Session management, logging, message formatting
- **`utils/chats.py`**: Standard LLM provider integration via LiteLLM
- **`utils/foundry.py`**: Azure AI Foundry agents with advanced capabilities

### Adding New Providers

1. Update `llm_config.json` with new model configuration
2. Ensure `get_llm_params()` in `chats.py` handles provider-specific parameters
3. Test the new provider through the chat interface

### Testing

Run the test suite to ensure functionality:

```bash
python -m pytest tests/
```

## 🔒 Security & Compliance

### BSP-Specific Features
- **Confidentiality**: Secure handling of sensitive BSP information
- **Compliance**: Adherence to BSP policies and regulations
- **Professional Tone**: Formal communication standards
- **Internal Use**: Designed exclusively for BSP employees

### Security Measures
- Header-based authentication via Azure App Service
- Secure session management with user context
- Encrypted communication with LLM providers
- Comprehensive audit logging

## 📈 Production Deployment

### Azure App Service Deployment

1. **Configure authentication** in Azure App Service
2. **Set environment variables** in App Service configuration
3. **Deploy application** using Azure DevOps or GitHub Actions
4. **Configure custom domain** and SSL certificates

### Environment Configuration

```bash
# Production environment variables
LLM_CONFIG='[{"model_deployment":"foundry/gpt-4o",...}]'
CHAINLIT_AUTH_SECRET="your-secret-key"
AIPROJECT_CONNECTION_STRING="your-connection-string"
```

## 🧪 Testing & Debugging

### Logging System
- **Console**: Color-coded logs with session/user context
- **File**: Rotating logs in `logs/app.log` (10MB, 30-day retention)
- **Format**: `[timestamp] [level] [session_id] [user_id] [location] message`

### Model Testing
Switch between providers via Chainlit chat profiles without code changes.

### Debug Mode
Enable debug logging by setting log level to `DEBUG` in `utils.py`.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow code standards (type hints, docstrings, formatting)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Code Standards

- **Type Hints**: Required for all function parameters and return values
- **Docstrings**: Required for all functions and classes
- **Formatting**: Follow existing code style and patterns
- **Testing**: Add tests for new functionality

## 📚 API Reference

### Core Functions

#### `chat_completion(messages: list) -> str`
Handles standard LLM provider interactions via LiteLLM.

#### `chat_agent(user_input: str) -> str`
Manages Azure AI Foundry agent interactions with advanced capabilities.

#### `append_message(role: str, content: str, elements: list) -> list`
Appends messages to chat history with file processing.

#### `init_settings() -> dict`
Initializes chat settings with BSP-specific instructions.

## 🐛 Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Verify Azure App Service authentication configuration
   - Check header values in development environment

2. **Model Not Responding**
   - Verify API keys in environment variables
   - Check model configuration in `llm_config.json`
   - Review network connectivity to provider endpoints

3. **File Upload Issues**
   - Ensure sufficient disk space for temporary files
   - Check file format compatibility with MarkItDown
   - Verify Azure AI Foundry configuration for advanced processing

### Debug Steps

1. Check application logs in `logs/app.log`
2. Verify environment variable configuration
3. Test individual provider endpoints
4. Review Chainlit console output for detailed errors

## 📄 License

This project is proprietary software developed for Bangko Sentral ng Pilipinas internal use.

## 📞 Support

For technical support and questions, please contact the BSP IT development team.

---

**BSP AI Assistant** - Empowering productivity through intelligent conversation