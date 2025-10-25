# thinking/dynamic_config.py
# Configuration for dynamic LLM-powered thinking

from typing import Optional

# Enable dynamic LLM-generated thinking steps
USE_DYNAMIC_THINKING = True

# Fallback to static patterns if LLM generation fails
FALLBACK_TO_STATIC = True

# Use caching for similar queries (improves performance)
ENABLE_THINKING_CACHE = True
CACHE_MAX_SIZE = 50

# Token budget for reasoning generation (keep low for speed)
REASONING_MAX_TOKENS = 300

# Thinking generation strategies
THINKING_STRATEGY = "llm"  # Options: "llm", "hybrid", "static"
# - "llm": Always use LLM to generate steps
# - "hybrid": Use LLM for complex queries, static for simple ones
# - "static": Use predefined patterns (original behavior)

# Complexity threshold for hybrid mode
HYBRID_COMPLEXITY_THRESHOLD = 30  # chars

# Model to use for reasoning generation (None = use default)
REASONING_MODEL: Optional[str] = None  # e.g., "gpt-4o-mini" for speed

# Temperature for reasoning generation (lower = more focused)
REASONING_TEMPERATURE = 0.3

# System prompt customization
REASONING_SYSTEM_PROMPT = """You are an AI reasoning analyzer. Generate brief, natural thinking steps that show transparent reasoning without giving away the answer."""

# Enable streaming updates (show steps as they're generated)
ENABLE_STREAMING_THINKING = True

# Timing adjustments for dynamic thinking
DYNAMIC_TIMING_CONFIG = {
    "initial_delay": 0.3,
    "step_delay": 0.4,
    "completion_delay": 0.3,
    "llm_generation_timeout": 5.0  # Max seconds to wait for LLM
}

# Minimum query length to use LLM generation
MIN_QUERY_LENGTH_FOR_LLM = 5

# Skip thinking for very short/simple queries
SKIP_THINKING_FOR_SIMPLE = True
SIMPLE_QUERY_KEYWORDS = ["hi", "hello", "thanks", "ok", "yes", "no"]


def should_use_dynamic_thinking(query: str, strategy: str = None) -> bool:
    """
    Determine if dynamic LLM thinking should be used.
    
    Args:
        query: User's input query
        strategy: Override strategy (optional)
        
    Returns:
        True if should use dynamic thinking
    """
    if not USE_DYNAMIC_THINKING:
        return False
    
    strategy = strategy or THINKING_STRATEGY
    
    # Check if query is too simple
    if SKIP_THINKING_FOR_SIMPLE:
        query_lower = query.lower().strip()
        if query_lower in SIMPLE_QUERY_KEYWORDS or len(query) < 5:
            return False
    
    # Check minimum length
    if len(query.strip()) < MIN_QUERY_LENGTH_FOR_LLM:
        return False
    
    # Strategy-based decision
    if strategy == "llm":
        return True
    elif strategy == "hybrid":
        # Use LLM only for complex queries
        return len(query) >= HYBRID_COMPLEXITY_THRESHOLD
    else:  # static
        return False


def get_thinking_strategy(query: str) -> str:
    """
    Determine which thinking strategy to use for this query.
    
    Args:
        query: User's input query
        
    Returns:
        Strategy name: "llm", "static", or "skip"
    """
    if not should_use_dynamic_thinking(query):
        # Check if we should skip entirely or use static
        if SKIP_THINKING_FOR_SIMPLE and len(query) < 10:
            return "skip"
        return "static"
    
    return "llm"


# Advanced: Custom reasoning prompts for different query types
CUSTOM_REASONING_PROMPTS = {
    "code": """Generate reasoning steps for a coding-related query. Focus on:
- Understanding requirements
- Planning implementation
- Considering edge cases
Query: {query}""",
    
    "data": """Generate reasoning steps for a data analysis query. Focus on:
- Data understanding
- Analysis approach
- Insight extraction
Query: {query}""",
    
    "creative": """Generate reasoning steps for a creative task. Focus on:
- Creative direction
- Ideation process
- Content structure
Query: {query}"""
}


def get_custom_prompt(query: str) -> Optional[str]:
    """
    Get a custom reasoning prompt if query matches specific patterns.
    
    Args:
        query: User's input query
        
    Returns:
        Custom prompt or None
    """
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["code", "function", "program", "script"]):
        return CUSTOM_REASONING_PROMPTS.get("code")
    elif any(word in query_lower for word in ["data", "analyze", "statistics"]):
        return CUSTOM_REASONING_PROMPTS.get("data")
    elif any(word in query_lower for word in ["write", "create", "story", "poem"]):
        return CUSTOM_REASONING_PROMPTS.get("creative")
    
    return None


# Performance monitoring
ENABLE_PERFORMANCE_LOGGING = True

# Log thinking generation time and success rate
def log_thinking_performance(
    query_length: int,
    generation_time: float,
    success: bool,
    num_steps: int
):
    """Log performance metrics for thinking generation."""
    if ENABLE_PERFORMANCE_LOGGING:
        from utils.utils import get_logger
        logger = get_logger()
        logger.info(
            f"Thinking performance - Query len: {query_length}, "
            f"Time: {generation_time:.2f}s, Success: {success}, "
            f"Steps: {num_steps}"
        )