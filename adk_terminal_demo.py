# ==============================================================================
# IMPORTS & PREREQUISITES
# ==============================================================================
import os                           # Reads environment variables
import asyncio                      # Handles asynchronous event loops
from dotenv import load_dotenv      # Loads keys from local .env

# Google ADK building blocks:
# Runner: Orchestrates the conversation and tool calls
# InMemorySessionService: Retains chat context between turns
# types: Gemini message formatting
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Import the pre-configured root agent from adk_agent.py
from adk_agent import root_agent


# ==============================================================================
# 1. SETUP SESSION SERVICE & RUNNER
# ==============================================================================
load_dotenv()

APP_NAME = "adk_classroom_chat"
session_service = InMemorySessionService()

runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)


# ==============================================================================
# 2. CHAT HANDLER FUNCTION
# ==============================================================================
async def send_message(user_id: str, session_id: str, text: str) -> str:
    """Sends a single message to the ADK agent and returns its final answer."""
    # Ensure session exists in the memory store
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    if not session:
        await session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )

    # Format user input for Google ADK
    msg = types.Content(role="user", parts=[types.Part(text=text)])

    # Stream runner events and extract the final assistant text
    reply = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=msg
    ):
        if event.is_final_response() and event.content and event.content.parts:
            reply = "".join(p.text for p in event.content.parts if p.text)

    return reply


# ==============================================================================
# 3. INTERACTIVE TERMINAL LOOP (For Classroom Demos & Testing)
# ==============================================================================
async def interactive_terminal():
    user_id = "student_user"
    session_id = "live_terminal_session"

    print("\n" + "=" * 65)
    print("💬 Google ADK Interactive Chat Terminal")
    print("Type your message below. Try asking: 'Where is my order ORD-101?'")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 65 + "\n")

    while True:
        try:
            user_input = input("👤 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("\n👋 Goodbye! Session ended.\n")
                break

            response = await send_message(user_id, session_id, user_input)
            print(f"🤖 Agent: {response}\n")

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Session interrupted. Exiting.\n")
            break


if __name__ == "__main__":
    asyncio.run(interactive_terminal())
