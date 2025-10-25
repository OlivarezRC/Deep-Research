# LiteLLM-based chat completion functionality for BSP AI Assistant
# This file handles standard LLM provider interactions using LiteLLM
# Supporting multiple providers like Azure OpenAI, Gemini, and Perplexity

import time
import chainlit as cl
from loguru import logger
from litellm import completion
from utils.utils import get_llm_models


# Get LLM parameters
def get_llm_params(messages: list, use_tools = False) -> dict:
    """
    Build parameters for LLM API calls based on current session configuration.
    
    Constructs the parameter dictionary needed for LiteLLM completion calls,
    including model-specific settings, API keys, and optional tools.
    
    Args:
        messages: List of message objects to send to the LLM
        use_tools: Whether to include function calling tools (default: False)
        
    Returns:
        dict: Complete parameter dictionary for LiteLLM completion
    """
    # Get chat settings
    chat_settings = cl.user_session.get("chat_settings")
    chat_profile = cl.user_session.get("chat_profile")
    temperature = chat_settings.get("temperature")
    provider = chat_settings.get("model_provider")
    model_name = chat_settings.get("model_name")

    # Get the model details from the selected model
    llm_details = next((item for item in get_llm_models() if item["model_deployment"] == chat_profile), {})
    logger.debug(f"messages: {messages}")

    chat_parameters = {
        "model": chat_profile,
        "messages": messages,
        "stream": True,
    }
    
    # Only add api_key if it exists in llm_details
    if "api_key" in llm_details:
        chat_parameters["api_key"] = llm_details["api_key"]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web using SERP API",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    if provider == "azure":
        if llm_details.get("api_version"):
            chat_parameters["api_version"] = llm_details["api_version"]

        if llm_details.get("api_endpoint"):
            chat_parameters["api_base"] = llm_details["api_endpoint"]

        # Models that accept temperature
        if model_name not in ["o3-mini"]:
            chat_parameters["temperature"] = temperature
    else:
        chat_parameters["temperature"] = float(temperature)

    # Append search_web tool
    if use_tools:
        chat_parameters["tools"] = tools

    return chat_parameters


# Chat completion function
async def chat_completion(messages: list, use_tools = False) -> str:
    """
    Generate a response from the configured LLM provider using LiteLLM.
    
    Streams the response from the LLM and handles citations, thinking indicators,
    and error recovery. Works with multiple providers through LiteLLM.
    
    Args:
        messages: List of messages to send to the model
        use_tools: Whether to enable function calling tools (default: False)
    
    Returns:
        str: The generated response from the model
        
    Raises:
        RuntimeError: If response generation fails
    """
    try:
        # Get chat settings
        chat_settings = cl.user_session.get("chat_settings")
        model_name = chat_settings.get("model_name")

        # Show thinking message to user
        msg = await cl.Message(f"[{model_name}] thinking...", author="GPT Model").send()
        chat_parameters = get_llm_params(messages)
        logger.info(f"Chat parameters: {chat_parameters}")

        # Create chat completion
        response = completion(**chat_parameters)
        is_thinking = True
        last_chunk = None

        for chunk in response:
            # Check if the message is still thinking
            if is_thinking:
                msg.content = ""
                is_thinking = False
                logger.info(f"Elapsed time: {(time.time() - cl.user_session.get('start_time')):.2f} seconds")

            if chunk.choices and chunk.choices[0].delta.content:
                # await msg.stream_token(chunk.choices[0].delta.content)
                msg.content += chunk.choices[0].delta.content
                await msg.update()

            if "citations" in chunk:
                last_chunk = chunk

        logger.info("Checking for citations inside chat_completion.")
        if last_chunk and "citations" in last_chunk:
            msg.content += f"\n\n**Sources:**"

            # Loop through citations and append them to the response
            for citation in last_chunk.citations:
                msg.content += f"\n[{citation}]({citation})"



        logger.info(f"Last Chunk: {last_chunk}")

        # Remove the thinking message by splitting the content
        if msg and msg.content.startswith(f"<think>"):
            msg.content = msg.content.split(f"</think>")[-1].strip()

        await msg.update()
        return msg.content

    except Exception as e:
        raise RuntimeError(f"Error generating response in chat_completion: {str(e)}")
