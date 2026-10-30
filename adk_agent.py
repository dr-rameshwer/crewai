# ==============================================================================
# IMPORTS & CORE BUILDING BLOCKS
# ==============================================================================
import os                           # Reads environment variables (API keys) from the operating system
import asyncio                      # Handles asynchronous runtime execution
from dotenv import load_dotenv      # Loads key-value pairs from a local .env file into os.environ

# Google ADK (Agent Development Kit) core building blocks:
# Agent: Autonomous persona defined with instructions, a Gemini model, and custom tools
# Runner: Orchestrator engine coordinating the reasoning loop, tool execution, and state
# InMemorySessionService: State manager keeping multi-turn conversational memory per user & session
# types: Gemini data structures for message formatting (Content, Part)
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# ==============================================================================
# 1. ENVIRONMENT & CONFIGURATION
# ==============================================================================
load_dotenv()  # Searches for .env and loads GOOGLE_API_KEY into environment

# Verify that the Gemini API key is configured
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("⚠️ Missing GOOGLE_API_KEY. Please set it in your .env file.")

# Model identifier for Google Gemini
MODEL_NAME = "gemini-3.8-flash"


# ==============================================================================
# 2. DEFINING AGENT TOOLS (Standard Python Functions)
# ==============================================================================
# In Google ADK, tools are plain Python functions.
# CRITICAL RULES FOR ADK TOOLS:
# 1. Type Hints: Every parameter must have a type annotation (e.g. order_id: str).
# 2. Docstrings: Gemini reads the docstring to understand WHEN and HOW to call the tool.

def track_order(order_id: str) -> dict:
    """Retrieves live status, shipping carrier, and ETA for a customer order.

    Args:
        order_id: The unique tracking identifier (e.g., ORD-101, ORD-202).
    """
    mock_database = {
        "ORD-101": {
            "order_id": "ORD-101",
            "status": "In Transit",
            "carrier": "BlueDart",
            "eta": "Delivering tomorrow by 4:00 PM",
            "items": ["Wireless Mechanical Keyboard"],
        },
        "ORD-202": {
            "order_id": "ORD-202",
            "status": "Processing",
            "carrier": "FedEx",
            "eta": "Expected to ship in 24 hours",
            "items": ["USB-C Docking Station"],
        },
    }
    return mock_database.get(
        order_id.strip().upper(),
        {
            "order_id": order_id,
            "status": "Not Found",
            "detail": f"No active order found matching '{order_id}'. Please verify the ID.",
        },
    )


def cancel_order(order_id: str, reason: str = "Customer request") -> dict:
    """Cancels an existing customer order if it has not yet shipped.

    Args:
        order_id: The order identifier to cancel (e.g., ORD-101, ORD-202).
        reason: The reason provided by the user for cancellation.
    """
    clean_id = order_id.strip().upper()
    if clean_id == "ORD-202":
        return {
            "order_id": clean_id,
            "status": "Cancelled",
            "refund_status": "Initiated to original payment method",
            "reason": reason,
        }
    return {
        "order_id": clean_id,
        "status": "Cannot Cancel",
        "detail": "Order has already been dispatched. Please initiate a return upon delivery.",
    }


# ==============================================================================
# 3. DEFINING THE AUTONOMOUS AGENT
# ==============================================================================
# The Agent encapsulates persona instructions, the underlying LLM, and tools
root_agent = Agent(
    name="ecommerce_support_agent",                                         # Agent identifier for logging & routing
    model=MODEL_NAME,                                                       # Google Gemini model backing this agent
    instruction="""                                                         # System prompt guiding agent behavior
    You are an intelligent, empathetic customer service agent for an e-commerce platform.
    
    Guidelines:
    1. Always be polite, concise, and helpful.
    2. When a user asks about an order status, invoke `track_order`.
    3. When a user wants to cancel an order, invoke `cancel_order`.
    4. If the user has not provided an order ID, ask for it politely before taking action.
    5. Summarize tool results clearly in conversational language without exposing raw JSON errors.
    """,
    tools=[track_order, cancel_order]                                       # List of attached Python functions
)


# ==============================================================================
# 4. ORCHESTRATION & SESSION MEMORY
# ==============================================================================
# ADK manages multi-turn conversational state natively:
# - app_name: Namespaces the application
# - InMemorySessionService: Stores conversation turns per (app_name, user_id, session_id)
# - Runner: Coordinates the reasoning loop between Agent, Tools, and Session memory

APP_NAME = "ecommerce_support_app"
session_service = InMemorySessionService()

runner = Runner(
    agent=root_agent,                                                       # The agent handling incoming requests
    app_name=APP_NAME,                                                      # Namespace identifier
    session_service=session_service                                         # State & history manager
)


# ==============================================================================
# 5. CONVERSATION HELPER FUNCTION
# ==============================================================================
async def ask_agent(user_id: str, session_id: str, query: str) -> str:
    """Sends a user query through the ADK Runner and returns the finalized response."""
    # Ensure a session exists in memory for this user
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    if not session:
        await session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )

    # Wrap the string into Gemini Content format
    user_message = types.Content(role="user", parts=[types.Part(text=query)])

    # Stream runner events and extract the final response text
    agent_reply = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=user_message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            agent_reply = "".join(part.text for part in event.content.parts if part.text)

    return agent_reply


# ==============================================================================
# 6. EXECUTION ENTRY POINT (Run Standalone for Class / Testing)
# ==============================================================================
async def main():
    user = "student_alice"
    session = "demo_session_01"

    print("\n" + "=" * 60)
    print("🤖 GOOGLE ADK (Agent Development Kit) LIVE DEMO")
    print("=" * 60)

    # Turn 1: Query without order ID (Tests reasoning & missing info prompt)
    q1 = "Hi! Where is my order?"
    print(f"\n👤 [User]: {q1}")
    r1 = await ask_agent(user, session, q1)
    print(f"🤖 [Agent]: {r1}")

    # Turn 2: Providing order ID (Tests autonomous tool calling: track_order)
    q2 = "My order ID is ORD-101."
    print(f"\n👤 [User]: {q2}")
    r2 = await ask_agent(user, session, q2)
    print(f"🤖 [Agent]: {r2}")

    # Turn 3: Conversational follow-up (Tests session memory retention)
    q3 = "Can you cancel it instead?"
    print(f"\n👤 [User]: {q3}")
    r3 = await ask_agent(user, session, q3)
    print(f"🤖 [Agent]: {r3}")

    print("\n" + "=" * 60)
    print("🎉 All 3 turns completed successfully with Google ADK!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
