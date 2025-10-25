# Utility functions for BSP AI Assistant
# This file contains core utilities for session management, logging configuration,
# message formatting, model configuration, and chat settings initialization

import os, sys, json, base64, logging
import chainlit as cl
from loguru import logger
from dotenv import load_dotenv
from markitdown import MarkItDown
from chainlit.input_widget import Slider, TextInput

# Load environment variables
load_dotenv()
md = MarkItDown()

# Disable verbose connection logs
set_logging = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
set_logging.setLevel(logging.WARNING)


# Function to truncate long messages
def truncate(record):
    """
    Truncate long log messages to prevent excessive output.
    
    Args:
        record: Log record object containing the message
        
    Returns:
        bool: Always True to allow the record to pass through
    """
    message = record["message"]
    if len(message) > 1000:
        record["message"] = message[:1000] + "… [truncated]"
    return True  # Always return True to allow the record to pass through


# Function to add session and user context to log records
def add_context(record):
    """
    Add session and user context to log records for enhanced traceability.
    
    Extracts session ID and user information from Chainlit session
    and adds them to the log record's extra data.
    
    Args:
        record: Log record object to enhance with context
        
    Returns:
        bool: Always True to allow the record to pass through
    """
    # Set session context for logging
    session_id = cl.user_session.get("id", "unknown-session")
    app_user = cl.user_session.get("user")
    
    # Extract user ID from user object if available
    user_id = "anonymous"
    if app_user and hasattr(app_user, 'metadata') and app_user.metadata.get('id'):
        user_id = app_user.metadata['id']
    elif app_user and hasattr(app_user, 'identifier'):
        user_id = app_user.identifier

    # Add context to the record's extra data
    record["extra"]["session_id"] = session_id
    record["extra"]["user_id"] = user_id
    
    return True


# Enhanced format with proper level colors, cleaner layout, and session/user context
CONSOLE_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<magenta>{extra[session_id]} {extra[user_id]}</magenta> | "
    "<cyan>{name}:{function}:{line}</cyan> | "
    "<level>->>>>> {message}</level>"
)

# File format without colors but with session/user context
FILE_LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | "
    "{level: <8} | "
    "{extra[session_id]} {extra[user_id]} | "
    "{name}:{function}:{line} | "
    "->>>>> {message}"
)

# Remove default logger and configure custom logging
logger.remove()

# Add console sink with colors for terminal output
logger.add(
    sink=sys.stderr,
    format=CONSOLE_LOG_FORMAT,
    colorize=True,
    backtrace=True,
    diagnose=True,
    filter=lambda record: truncate(record) and add_context(record),
    level="DEBUG",  # Set to DEBUG for development, INFO for production
    enqueue=False  # Immediate output for real-time feedback
)

# Add file sink without colors for log file
logger.add(
    sink="logs/app.log",
    format=FILE_LOG_FORMAT,
    colorize=False,
    backtrace=True,
    diagnose=True,
    filter=lambda record: truncate(record) and add_context(record),
    level="INFO",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
    enqueue=False
)

# Expose logger
get_logger = lambda: logger


# Get llm models from llm_config.json
def get_llm_models() -> list:
    """
    Retrieve the list of available LLM models from the configuration.
    
    Loads model configurations either from environment variable (production)
    or from the configuration file (development).
    
    Returns:
        list: List of LLM model configuration dictionaries
    """
    parse_env = True

    if parse_env:
        try:
            llm_config_env = os.getenv("LLM_CONFIG")
            if llm_config_env and llm_config_env.strip():
                return json.loads(llm_config_env)
            else:
                # Fall back to file if env var is empty
                parse_env = False
        except (json.JSONDecodeError, Exception):
            # Fall back to file if parsing fails
            parse_env = False
    
    if not parse_env:
        with open("llm_config/llm_config.json", "r") as file:
            llm_config = json.load(file)

            # Copy this to the env file
            # logger.debug(json.dumps(llm_config).replace(" ", ""))
            return llm_config


# Get llm workweb models from llm_workweb.json or LLM_WORKWEB environment variable
def get_llm_workweb() -> list:
    """
    Retrieve the list of available LLM workweb models from the configuration.
    
    Loads model configurations either from environment variable (production)
    or from the configuration file (development).
    
    Returns:
        list: List of LLM workweb model configuration dictionaries
    """
    parse_env = True

    if parse_env:
        try:
            llm_workweb_env = os.getenv("LLM_WORKWEB")
            if llm_workweb_env and llm_workweb_env.strip():
                return json.loads(llm_workweb_env)
            else:
                # Fall back to file if env var is empty
                parse_env = False
        except (json.JSONDecodeError, Exception):
            # Fall back to file if parsing fails
            parse_env = False
    
    if not parse_env:
        with open("llm_config/llm_workweb.json", "r") as file:
            llm_workweb = json.load(file)

            # Copy this to the env file
            # logger.debug(json.dumps(llm_workweb).replace(" ", ""))
            return llm_workweb


# Append openai chat completion message
def append_message(role: str, content: str, elements: list = []) -> list:
    """
    Append a message to the chat history with proper formatting and file handling.
    
    Processes user messages with attachments, converts files to appropriate formats,
    and maintains chat history with automatic pruning.
    
    Args:
        role: The role of the message sender ('user' or 'assistant')
        content: The text content of the message
        elements: List of file attachments (default: empty list)
        
    Returns:
        list: Complete message list including system prompt and chat history
    """
    instructions = cl.user_session.get("chat_settings").get("instructions")

    # Create system message with instructions
    system_prompt = [{
        "role": "system",
        "content": [{"type": "text", "text": instructions}]
    }]

    chat_history = cl.user_session.get("chat_history", [])
    contents = [{"type": "text", "text": content}]
    file_contents = []
    file_uploads = []

    # Check if the role is assistant and add the images to the message
    if role == "user":
        for element in elements:
            logger.info(f"Uploaded file: {element}")
            # is_foundry = cl.user_session.get("chat_settings").get("model_provider") == "foundry"
            image_base64 = None

            # check if the element is an image
            if element.mime.startswith("image/"):
                encoded_image = base64.b64encode(open(element.path, 'rb').read()).decode('ascii')
                image_base64 = f"data:{element.mime};base64,{encoded_image}"
                contents.append({"type": "image_url", "image_url": { "url": image_base64}})

            # Convert the file to markdown format
            else:
                md_result = md.convert(element.path)
                file_contents.append(f"<file_name:{element.name}>{md_result.text_content}</file_name:{element.name}>")

            file_uploads.append({
                "name": element.name,
                "mime": element.mime,
                "path": element.path,
                "base64": image_base64
            })

    # Set file uploads in session
    cl.user_session.set("file_uploads", file_uploads)
    cl.user_session.set("file_contents", file_contents)

    # Check if there are any uploaded files and add them to the message
    if len(file_contents) > 0:
        contents.append({"type": "text", "text": "\n\n".join(file_contents)})

    logger.info(f"[{role}]: {contents}")
    # Add message to history
    chat_history.append({
        "role": role,
        "content": contents
    })

    # Prune chat history to keep only the 10 most recent messages
    if role == "assistant" and len(chat_history) > 10:
        chat_history = chat_history[-10:]

    # Update chat history in session
    cl.user_session.set("chat_history", chat_history)
    
    # Return combined messages (system message + chat history)
    return system_prompt + chat_history


# Initialize chat settings
async def init_settings() -> None:
    """
    Initialize chat settings with default instructions and UI controls.
    
    Creates the chat settings interface with temperature slider and
    instructions text input, pre-populated with BSP AI Assistant guidelines.
    
    Returns:
        dict: Chat settings dictionary containing user preferences
    """
    instructions = """
You are BSP AI Assistant, an advanced conversational AI model designed to assist internal employees of Bangko Sentral ng Pilipinas (BSP).
Your primary role is to provide accurate, timely, and relevant information, support productivity tasks, and enhance the overall efficiency of BSP operations.

### Personality Traits
- Professional: Maintain a formal and respectful tone, reflecting the standards of BSP.
- Knowledgeable: Provide accurate and up-to-date information on BSP policies, procedures, and financial regulations.
- Supportive: Offer assistance and solutions to employees' queries and tasks, promoting a collaborative work environment.
- Efficient: Deliver concise and clear responses to ensure quick and effective communication.

### Capabilities
- Information Retrieval: Access and provide information on BSP policies, procedures, financial regulations, and internal guidelines.
- Task Assistance: Help with scheduling, document management, and other productivity-related tasks.
- Problem Solving: Offer solutions to common issues faced by employees, including technical support and procedural clarifications.
- Learning and Adaptation: Continuously learn from interactions to improve responses and adapt to the evolving needs of BSP employees.

### Safety Guidelines
- Confidentiality: Ensure the privacy and security of sensitive information. Do not share confidential data outside the scope of internal BSP operations.
- Accuracy: Provide correct and verified information. If unsure, indicate the need for further verification or direct the employee to the appropriate department.
- Compliance: Adhere to BSP's internal policies and regulations. Avoid sharing information that conflicts with BSP's standards or legal requirements.
- Transparency: Inform employees if a request exceeds your capabilities or does not align with safety guidelines. Maintain a respectful and professional demeanor.

### Interaction Style
- Formal and Respectful: Use language that reflects the professional environment of BSP.
- Concise and Clear: Ensure responses are straightforward and easy to understand.
- Helpful and Supportive: Aim to assist employees in resolving their queries and completing tasks efficiently.
"""
    settings = await cl.ChatSettings(
        [
            Slider(
                id="temperature",
                label="Temperature",
                initial=0.7,
                min=0,
                max=2,
                step=0.1,
            ),
            TextInput(
                id="instructions",
                label="Instructions",
                initial=instructions
            ),
        ]
    ).send()

    return settings


# Get llm details from the selected model
def get_llm_details() -> dict:
    """
    Retrieve the configuration details of the currently selected LLM model.
    
    Extracts model details from the configuration based on the active
    chat profile and updates session with model provider information.
    
    Returns:
        dict: Model configuration details including API endpoints and keys
    """
    chat_settings = cl.user_session.get("chat_settings")
    provider, model_name = cl.user_session.get("chat_profile").split("/")

    # Set the model name and provider in the session
    chat_settings["model_name"] = model_name
    chat_settings["model_provider"] = provider
    cl.user_session.set("chat_settings", chat_settings)

    llm_details = next((item for item in get_llm_models() if item["model_deployment"].endswith(f"/{model_name}")), {})
    return llm_details
