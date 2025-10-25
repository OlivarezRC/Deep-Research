import time
import chainlit as cl
from utils.utils import (
    append_message, init_settings, get_llm_details, get_llm_models, get_logger,
)
from typing import Dict, Optional
from azure.ai.agents import AgentsClient
from azure.identity import DefaultAzureCredential
from utils.chats import chat_completion
from utils.foundry import chat_agent
from deep_research.pipeline import run_deep_research

# Import dynamic thinking modules
from thinking.dynamic_thinking import dynamic_thinking, cached_dynamic_thinking
from thinking.dynamic_config import (
    should_use_dynamic_thinking, get_thinking_strategy, 
    USE_DYNAMIC_THINKING
)

# Import analytics modules
from analytics.analytics_handler import handle_analytics_command, analytics_handler

logger = get_logger()


@cl.action_callback("set_mode")
async def set_mode(action: cl.Action):
    """
    Handle mode setting from work/web toggle switch.
    
    Args:
        action: The action object containing the mode payload
    """
    # Sanitize input
    value = str(action.payload.get("mode", "work")).lower()
    if value not in {"work", "web"}:
        value = "null1"

    # Persist to the per-session store
    cl.user_session.set("mode", value)
    logger.info(f"Mode set to: {value}")


@cl.header_auth_callback
def header_auth_callback(headers: Dict) -> Optional[cl.User]:
    """
    Handle authentication using headers from Azure App Service.
    
    Extracts user information from HTTP headers for authentication
    in Azure App Service environments.
    
    Args:
        headers: Dictionary containing HTTP request headers
        
    Returns:
        Optional[cl.User]: User object if authentication successful, None otherwise
    """
    # Verify the signature of a token in the header (ex: jwt token)
    # or check that the value is matching a row from your database
    user_name = headers.get('X-MS-CLIENT-PRINCIPAL-NAME', 'dummy@microsoft.com')
    user_id = headers.get('X-MS-CLIENT-PRINCIPAL-ID', '9876543210')
    logger.debug(f"Auth Headers: {headers}")

    if user_name:
        return cl.User(identifier=user_name, metadata={"role": "admin", "provider": "header", "id": user_id})
    else:
        return None


@cl.set_chat_profiles
async def chat_profile():
    """
    Set up available chat profiles based on configured LLM models.
    
    Creates chat profiles from the model configurations, allowing users
    to select different language models for their conversations.
    
    Returns:
        List[cl.ChatProfile]: List of available chat profiles
    """
    llm_models = get_llm_models()
    # get a list of model names from llm_models
    model_list = [f"{model["model_deployment"]}--{model["description"]}" for model in llm_models]
    profiles = []

    for item in model_list:
        model_deployment, description = item.split("--")

        # Create a profile for each model
        profiles.append(
            cl.ChatProfile(
                name=model_deployment,
                markdown_description=description
            )
        )

    return profiles


@cl.set_starters
async def set_starters():
    """
    Define starter conversation prompts for the chat interface.
    
    Provides pre-configured conversation starters to help users
    begin interactions with the AI assistant.
    
    Returns:
        List[cl.Starter]: List of starter conversation prompts
    """
    return [
        cl.Starter(
            label="Morning routine ideation",
            message="Can you help me create a personalized morning routine that would help increase my productivity throughout the day? Start by asking me about my current habits and what activities energize me in the morning.",
            icon="/public/bulb.webp",
            ),

        cl.Starter(
            label="Spot the errors",
            message="How can I avoid common mistakes when proofreading my work?",
            icon="/public/warning.webp",
            ),
        cl.Starter(
            label="Get more done",
            message="How can I improve my productivity during remote work?",
            icon="/public/rocket.png",
            ),
        cl.Starter(
            label="Boost your knowledge",
            message="Help me learn about [topic]",
            icon="/public/book.png",
            ),
        cl.Starter(
            label="Analyze data",
            message="/analytics both - Upload a CSV or Excel file to get charts and insights",
            icon="/public/chart.png",
            )
        ]


@cl.on_chat_resume
async def on_chat_resume(thread):
    """
    Handle chat resumption when a user returns to an existing conversation.
    
    Args:
        thread: The conversation thread being resumed
    """
    pass


@cl.on_chat_start
async def start():
    """
    Initialize the chat session and send a welcome message.
    
    Sets up chat settings, initializes Azure AI Foundry agents if needed,
    and prepares the conversation environment for the user.
    """
    try:
        cl.user_session.set("chat_settings", await init_settings())
        llm_details = get_llm_details()

        # Try to render the bridge element
        try:
            bridge = cl.CustomElement(name="SettingsBridge", props={}, display="inline")
            msg = cl.Message(content="How can I help you today?", author="LikhAI", elements=[bridge])
            await msg.send()
        except Exception as e:
            raise RuntimeError(f"Error on chat start: {str(e)}")

        # Create an instance of the AgentsClient using DefaultAzureCredential
        if cl.user_session.get("chat_settings").get("model_provider") == "foundry" and not cl.user_session.get("thread_id"):
            agents_client = AgentsClient(
                # conn_str=llm_details["api_key"],
                endpoint=llm_details["api_endpoint"],
                credential=DefaultAzureCredential()
            )

            # Create a thread for the agent
            thread = agents_client.threads.create()
            cl.user_session.set("thread_id", thread.id)
            logger.info(f"New thread created, thread ID: {thread.id}")

    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}", author="Error").send()
        logger.error(f"Error: {str(e)}")


@cl.on_message
async def on_message(message: cl.Message):
    user_input = message.content.strip()
    mode = cl.user_session.get("mode", "default")
    analytics_mode = cl.user_session.get("analytics_mode", False)

    # Route to analytics if user typed the command
    if user_input.lower().startswith("/analytics"):
        await handle_analytics_command(user_input, message.elements)
        return
    
    # If in analytics mode and not a command, treat as follow-up question about the data
    if analytics_mode and not user_input.startswith("/"):
        # Get stored data info
        data_info = cl.user_session.get("analytics_data", {})
        
        # Add context about the data to the prompt
        context_prompt = f"""[Analytics Context] 
        The user is asking about data with:
        - Shape: {data_info.get('shape', 'unknown')}
        - Columns: {', '.join(data_info.get('columns', []))}

        User question: {user_input}

        Please answer based on the previous analysis and data context."""

        # Use Foundry agent with context
        try:
        #    from utils.foundry import chat_agent
            
            msgs = append_message("user", context_prompt, message.elements)
         #   response = await chat_agent(context_prompt)
            
            if response:
                append_message("assistant", response)
                # Response already sent by chat_agent in Foundry mode
            else:
                await cl.Message(content="I don't have enough information to answer that. Could you clarify?").send()
            
            return
        except Exception as e:
            logger.error(f"Error in analytics follow-up: {e}")
            # Fall through to normal chat

    # Route to deep research if user typed a command or UI set the mode
    is_research_cmd = user_input.lower().startswith("/research ")
    if is_research_cmd or mode == "deep_research":
        topic = user_input[len("/research "):].strip() if is_research_cmd else user_input

        thinking_box = await cl.Message(author="🧠", content="(thinking…)").send()
        progress_box = await cl.Message(content="Starting research…").send()

        async def notify(event: str, data: dict):
            if event == "thinking":
                thinking_box.content = data.get("thoughts", "")
                await thinking_box.update()

            elif event == "generate_query":
                await cl.Message(
                    content=f"🔎 **Query:** {data['query']}\n\n_Why:_ {data.get('rationale','')}"
                ).send()

            elif event == "web_research":
                await cl.Message(
                    content=f"🌐 Collected {len(data.get('sources', []))} source(s)."
                ).send()

            elif event == "summarize":
                progress_box.content = "📝 Updating summary…"
                await progress_box.update()

            elif event == "reflection":
                await cl.Message(
                    content=f"🧭 Follow-up query: {data.get('query','')}"
                ).send()

            elif event == "routing":
                await cl.Message(
                    content=f"🔁 Decision: {data['decision']} (loop {data['loop_count']})"
                ).send()

            elif event == "finalize":
                imgs = data.get("images", [])
                elements = [
                    cl.Image(name=f"image-{i+1}", url=u, display="inline")
                    for i, u in enumerate(imgs)
                ]
                await cl.Message(
                    content=data["summary"],
                    elements=elements
                ).send()

        try:
            final_md = await run_deep_research(topic, notify=notify)
            if final_md:
                await cl.Message(content="✅ Research complete. See final summary above.").send()
        except Exception as e:
            await cl.Message(content=f"❌ Research error: {e}").send()
        return

    # ---------- normal chat path with DYNAMIC thinking ----------
    try:
        cl.user_session.set("start_time", time.time())

        provider = cl.user_session.get("chat_settings", {}).get("model_provider", "litellm")
        strategy = get_thinking_strategy(user_input)

        # Always show the dynamic thinking if enabled (Foundry included)
        if strategy == "llm" and USE_DYNAMIC_THINKING:
            logger.info(f"Using dynamic LLM thinking for query: {user_input[:50]}...")
            await cached_dynamic_thinking(user_input, provider, cache_enabled=True)

        # Process the message
        msgs = append_message("user", user_input, message.elements)

        if provider == "foundry":
            # Foundry streams & updates its own Chainlit message.
            full_response = await chat_agent(user_input)  # returns text but ALREADY rendered
            if not full_response:
                # Failsafe: if Foundry returned empty, send a minimal message so the UI isn't blank.
                await cl.Message(content="(No response received from the agent)").send()
            else:
                append_message("assistant", full_response)
            # DO NOT cl.Message(...).send() again here — avoids duplicate
        else:
            # Non-Foundry providers: we render once here.
            full_response = await chat_completion(msgs)
            append_message("assistant", full_response)
            await cl.Message(content=full_response).send()

    except Exception as e:
        await cl.Message(content=f"An error occurred: {e}", author="Error").send()
        logger.error(f"Error in on_message: {e}")