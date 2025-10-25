# thinking/dynamic_thinking.py
# Dynamic thinking process that uses LLM to generate reasoning steps

import chainlit as cl
from typing import List, Dict, Optional
from utils.utils import get_logger
import json
import re

logger = get_logger()


# Reasoning prompt - LLM generates markdown directly
REASONING_PROMPT = """

User Query: {query}
AI Output Response:
Generate your reasoning steps in clean markdown format. Use this structure:

**Step Name**
Brief description of what you're thinking/doing in this step (1-2 sentences). 

Keep steps concise, natural, and focused on the thinking process, not the answer itself.

Example:
Add a small header before giving reasoning process: *AI Reasoning...*
**Understanding the Core Question**
Analyzing the user's intent and identifying what specific information they're seeking.

**Gathering Relevant Context**
Accessing and organizing the necessary background knowledge for this topic.

**Structuring the Explanation**
Planning how to present the information in a clear, logical sequence.

Now generate reasoning steps for the user's query. Return ONLY the markdown formatted steps, nothing else.

"""


async def generate_thinking_with_llm(
    query: str, 
    provider: str = "foundry",
    model: Optional[str] = None
) -> str:
    """
    Use LLM to dynamically generate reasoning in markdown format.
    Returns the raw markdown response directly - no parsing needed.
    
    Args:
        query: User's input query
        provider: LLM provider (litellm or foundry)
        model: Specific model to use (optional)
        
    Returns:
        Markdown formatted reasoning steps
    """
    try:
        # Import based on provider
        if provider == "foundry":
            from utils.foundry import chat_agent
            # Get markdown response directly
            markdown_response = await chat_agent(REASONING_PROMPT.format(query=query))
        else:
            from utils.chats import chat_completion
            # Create a lightweight message for reasoning
            reasoning_messages = [
                {"role": "system", "content": "You are a reasoning step generator. Return clean markdown formatted thinking steps."},
                {"role": "user", "content": REASONING_PROMPT.format(query=query)}
            ]
            # Get markdown response directly
            markdown_response = await chat_completion(reasoning_messages)
        
        # Clean up the response (remove any extra whitespace)
        markdown_response = markdown_response.strip()
        
        if not markdown_response or len(markdown_response) < 10:
            # Fallback if response is too short or empty
            logger.warning("LLM response too short, using fallback")
            return get_fallback_markdown(query)
        
        logger.info(f"Successfully generated markdown reasoning via LLM")
        return markdown_response
        
    except Exception as e:
        logger.error(f"Error generating thinking with LLM: {e}")
        return get_fallback_markdown(query)


def get_fallback_markdown(query: str) -> str:
    """
    Generate simple fallback markdown if LLM fails.
    
    Args:
        query: User's input query
        
    Returns:
        Markdown formatted fallback reasoning
    """
    query_lower = query.lower()
    
    # Simple heuristic-based fallback
    if any(word in query_lower for word in ["how", "why", "explain"]):
        return """**Understanding the Question**
Analyzing your question and identifying the key information you're seeking.

**Gathering Relevant Knowledge**
Accessing the necessary background information and context for this topic.

**Structuring the Explanation**
Organizing the information in a clear, logical way to best answer your question."""
    
    elif any(word in query_lower for word in ["write", "create", "generate"]):
        return """**Understanding Requirements**
Analyzing what you need to create and any specific requirements.

**Planning the Approach**
Determining the best structure and content for what you're asking.

**Preparing the Output**
Getting ready to generate the content you've requested."""
    
    else:
        return """**Processing Your Request**
Understanding what you're asking and what would be most helpful.

**Formulating Response**
Preparing a clear and useful answer to your query."""


async def dynamic_thinking(
    query: str, 
    provider: str = "foundry",
    model: Optional[str] = None,
    use_llm: bool = True
) -> None:
    """
    Create a dynamic thinking process using LLM-generated markdown reasoning.
    Displays the markdown directly without any formatting.
    
    Args:
        query: User's input query
        provider: LLM provider
        model: Specific model to use
        use_llm: Whether to use LLM for generation (vs fallback)
    """
    # Generate markdown reasoning
    if use_llm:
        markdown_content = await generate_thinking_with_llm(query, provider, model)
    else:
        markdown_content = get_fallback_markdown(query)
    
    # Display the markdown directly as a message
    # thinking_msg = cl.Message(
    #     author="🧠 Thinking",
    #     content=markdown_content
    # )
    # await thinking_msg.send()


async def stream_thinking_updates(
    query: str,
    provider: str = "foundry"
) -> None:
    """
    Stream thinking updates - generates and displays markdown.
    
    Args:
        query: User's input query
        provider: LLM provider
    """
    try:
        # Generate markdown reasoning
        markdown_content = await generate_thinking_with_llm(query, provider)
        
        # Display it
        # thinking_msg = cl.Message(
        #     author="🧠 Thinking",
        #     content=markdown_content
        # )
        # await thinking_msg.send()
        
    except Exception as e:
        logger.error(f"Error in stream_thinking_updates: {e}")
        # Show fallback
        fallback = get_fallback_markdown(query)
        thinking_msg = cl.Message(
            author="🧠 Thinking",
            content=fallback
        )
        await thinking_msg.send()


# Cache for similar queries to avoid regenerating thinking
_thinking_cache: Dict[str, str] = {}  # Now stores markdown strings
_cache_max_size = 50


async def cached_dynamic_thinking(
    query: str,
    provider: str = "foundry",
    cache_enabled: bool = True
) -> None:
    """
    Dynamic thinking with caching for similar queries.
    Displays LLM-generated markdown directly.
    
    Args:
        query: User's input query
        provider: LLM provider
        cache_enabled: Whether to use caching
    """
    # Check cache for similar query
    cache_key = query.lower().strip()[:100]  # Simple cache key
    
    if cache_enabled and cache_key in _thinking_cache:
        logger.info("Using cached thinking markdown")
        markdown_content = _thinking_cache[cache_key]
    else:
        # Generate new markdown
        markdown_content = await generate_thinking_with_llm(query, provider)
        
        # Cache the markdown
        if cache_enabled and markdown_content:
            _thinking_cache[cache_key] = markdown_content
            
            # Maintain cache size
            if len(_thinking_cache) > _cache_max_size:
                # Remove oldest entry
                oldest_key = next(iter(_thinking_cache))
                del _thinking_cache[oldest_key]
    
    # Display the markdown directly
    # thinking_msg = cl.Message(
    #     author="🧠 Thinking",
    #     content=markdown_content
    # )
    # await thinking_msg.send()


def clear_thinking_cache():
    """Clear the thinking steps cache."""
    global _thinking_cache
    _thinking_cache.clear()
    logger.info("Thinking cache cleared")