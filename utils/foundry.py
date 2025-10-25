# Azure AI Foundry agent integration for BSP AI Assistant
# This file handles interactions with Azure AI Foundry agents
# Supporting advanced capabilities like code interpretation and file processing

import time
import chainlit as cl
from typing import List
from pathlib import Path
from loguru import logger
from azure.ai.agents import AgentsClient
from azure.identity import DefaultAzureCredential
from utils.utils import get_llm_models, get_llm_workweb
from azure.ai.agents.models import (
    CodeInterpreterTool,
    MessageAttachment,
    MessageInputContentBlock,
    MessageInputTextBlock,
    MessageImageFileParam,
    MessageInputImageFileBlock,
    FilePurpose,
    MessageRole,
    AgentStreamEvent,
    MessageDeltaChunk,
    ThreadRun,
)


# Chat with Azure AI Agents
async def chat_agent(user_input: str) -> str:
    """
    Generate a response from Azure AI Foundry agents with advanced capabilities.
    
    Handles file uploads, code interpretation, and streaming responses from
    Azure AI Foundry agents. Supports image generation and file annotations.
    
    Args:
        user_input: The user's message to send to the agent
        
    Returns:
        str: The agent's complete response text
        
    Raises:
        RuntimeError: If agent response generation fails
    """
    try:
        # Get chat settings
        chat_settings = cl.user_session.get("chat_settings")
        chat_profile = cl.user_session.get("chat_profile")
        model_name = chat_settings.get("model_name")        # Get the model details from the selected model
        llm_details = next((item for item in get_llm_models() if item["model_deployment"] == chat_profile), {})
        
        # Get the model_id from llm_workweb by mapping model_deployment and mode
        mode = cl.user_session.get("mode")
        model_id = None
        if llm_details and mode:
            llm_workweb = get_llm_workweb()
            workweb_model = next((item for item in llm_workweb 
                                if item["model_deployment"] == llm_details["model_deployment"] 
                                and item["mode"] == mode), {})
            model_id = workweb_model.get("model_id")

        logger.debug(f"Mapped model_id: {model_id} for deployment: {llm_details['model_deployment']} and mode: {mode}")
        
        # Check if model_id is None and raise error
        if model_id is None:
            raise RuntimeError("Please refresh this page.")
        
        is_first_message = not cl.user_session.get("first_message")
        if is_first_message:
            cl.user_session.set("first_message", user_input)

        # Show thinking message to user
        msg = await cl.Message(f"[{model_name}] thinking...", author="GPT Model").send()
        if not msg:
            raise Exception("Failed to create message object")
        
        # Attach ThreadNameUpdater element if this is the first message
        if is_first_message:
            # Create a custom element that will trigger the thread name update
            thread_name_updater = cl.CustomElement(
                name="ThreadNameUpdater", 
                props={"userInput": user_input}, 
                display="inline"
            )
            msg.elements = [thread_name_updater]
            await msg.update()

        # Create an instance of the AgentsClient using DefaultAzureCredential
        agents_client = AgentsClient(
            endpoint=llm_details["api_endpoint"],
            credential=DefaultAzureCredential()
        )

        thread_id = cl.user_session.get("thread_id")
        file_uploads = cl.user_session.get("file_uploads", [])
        file_contents = cl.user_session.get("file_contents", [])
        # content_blocks = user_input
        attachments = []
        content_blocks = [MessageInputTextBlock(text=user_input)]

        # Loop through file contents to append to content blocks
        for content in file_contents:
            content_blocks.append(MessageInputTextBlock(text=content))

        # Loop through file uploads to prepare content blocks and attachments
        for upload in file_uploads:
            logger.info(f"File upload: {upload}")

            # Upload a file and wait for it to be processed
            if upload["path"]:
                file = agents_client.files.upload_and_poll(
                    file_path=upload["path"], purpose=FilePurpose.AGENTS
                )
                logger.info(f"File ID: {file.id}")

                # Create a message with the attachment
                attachment = MessageAttachment(file_id=file.id, tools=CodeInterpreterTool().definitions)
                attachments.append(attachment)

                # If the file is an image, create a content block for it
                if upload["mime"].startswith("image/"):
                    file_param = MessageImageFileParam(file_id=file.id, detail="high")
                    content_blocks: List[MessageInputContentBlock] = [
                        MessageInputTextBlock(text=user_input),
                        MessageInputImageFileBlock(image_file=file_param),
                    ]

        logger.debug(f"Content blocks: {content_blocks}")
        logger.debug(f"Attachments: {attachments}")

        # Create a message, with the prompt being the message content that is sent to the model
        agents_client.messages.create(
            thread_id=thread_id,
            role="user",
            content=content_blocks,
            attachments=attachments
        )

        is_thinking = True        # Run the agent to process tne message in the thread
        with agents_client.runs.stream(thread_id=thread_id, agent_id=model_id) as stream:
            msg.content = ""
            # Delete msg.elements
            msg.elements = []
            for event_type, event_data, _ in stream:
                if isinstance(event_data, MessageDeltaChunk):
                    msg.content += event_data.text
                    if msg:
                        await msg.update()

                    if is_thinking:
                        logger.info(f"Elapsed time: {(time.time() - cl.user_session.get('start_time')):.2f} seconds")
                        is_thinking = False

                elif isinstance(event_data, ThreadRun):
                    if event_data.status == "failed":
                        logger.error(f"Run failed. Error: {event_data.last_error}")
                        raise Exception(event_data.last_error)

                elif event_type == AgentStreamEvent.ERROR:
                    logger.error(f"An error occurred. Data: {event_data}")
                    raise Exception(event_data)

        # Get all messages from the thread
        messages = agents_client.messages.list(thread_id)
        images = []

        # Process the messages to extract image contents and file path annotations
        for message in messages:
            last_image = None
            # Save every image file in the message
            if message.image_contents:
                last_image = message.image_contents[-1]
                logger.info(f"Response message: {message}")

            if last_image and "file_id" in last_image:
                # If the last image has a file_id, save it to the current working directory
                file_id = last_image.file_id
                file_name = f"{file_id}_image_file.png"
                agents_client.files.save(file_id=file_id, file_name=file_name)
                image = cl.Image(path=f"{Path.cwd() / file_name}", name=file_name, display="inline")
                images.append(image)

        # Append the images to the message
        if len(images) > 0:
            msg.elements = images        # Get the last message from the agent

        response_message = agents_client.messages.get_last_message_text_by_role(thread_id=thread_id, role=MessageRole.AGENT)
        if not response_message:
            raise Exception("No response from the model.")

        msg.content = response_message.text.value

        # Define a list of known "bad" domains to filter out
        local_or_dev_domains = [
            ".app.github.dev",
            "localhost",
            "127.0.0.1",
            "10.0.0.1",
            "192.168.0.1",
        ]
        
        valid_citations = []
        for annotation in response_message.text.annotations:
            logger.info(f"Checking Annotation: {annotation}")
            
            # Ensure the annotation has a url_citation and a valid URL
            if annotation.url_citation.url.startswith("http://") or annotation.url_citation.url.startswith("https://"):
                citation_url = annotation.url_citation.url
                logger.info(f"citation_url: {citation_url}")
                
                try:
                    valid_citations.append(f"[{annotation.url_citation.title}]({citation_url})")

                except Exception as e:
                    logger.warning(f"Failed to parse URL: {citation_url}, Error: {e}")
        
        # Now, append the filtered citations to the message content
        if valid_citations:
            msg.content += f"\n\n**Sources:**"
            for citation in valid_citations:
                msg.content += f"\n{citation}"

        if msg:
            await msg.update()
        return msg.content

    except Exception as e:
        raise RuntimeError(f"Error generating response in chat_agent: {str(e)}")