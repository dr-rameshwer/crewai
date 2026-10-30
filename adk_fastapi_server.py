# ==============================================================================
# IMPORTS & ARCHITECTURAL OVERVIEW
# ==============================================================================
# In production, AI agents are exposed as web APIs so mobile apps, React frontends,
# or internal microservices can chat with them over standard HTTP REST requests.
import os                           # Reads environment variables
from contextlib import asynccontextmanager  # Manages FastAPI startup & shutdown lifecycles
from uuid import uuid4              # Generates unique session IDs when clients don't provide one

from dotenv import load_dotenv      # Loads .env configuration
from fastapi import FastAPI, HTTPException  # High-performance async web framework
from pydantic import BaseModel, Field       # Data validation and request/response schemas

# Google ADK building blocks:
# Runner: Executes the agent logic per incoming HTTP request
# InMemorySessionService: Retains conversation memory across separate HTTP calls
# types: Gemini Content/Part message schemas
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Import the root agent configured with custom tools from adk_agent.py
from adk_agent import root_agent


# ==============================================================================
# 1. ENVIRONMENT CONFIGURATION
# ==============================================================================
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("⚠️ Missing GOOGLE_API_KEY. Please add it to your .env file.")


# ==============================================================================
# 2. ADK RUNNER & SESSION SETUP
# ==============================================================================
# The runner and session service are instantiated at the application level.
# This ensures conversation history persists across multiple HTTP POST requests.
APP_NAME = "ecommerce_order_support_api"
session_service = InMemorySessionService()

runner = Runner(
    agent=root_agent,                                                       # Autonomous agent with tools
    app_name=APP_NAME,                                                      # Namespace identifier
    session_service=session_service                                         # State & conversation memory store
)


# ==============================================================================
# 3. FASTAPI LIFECYCLE (LIFESPAN)
# ==============================================================================
# Ensures clean resource teardown when the server stops
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await runner.close()                                                    # Gracefully close ADK runner


# ==============================================================================
# 4. INITIALIZE FASTAPI APPLICATION
# ==============================================================================
app = FastAPI(
    title="Google ADK Order Support API",
    description="Production REST API powered by Google ADK Agent & Custom Tools.",
    version="1.0.0",
    lifespan=lifespan,
)


# ==============================================================================
# 5. DATA SCHEMAS (Request & Response Validation)
# ==============================================================================
class UserQuery(BaseModel):
    """Schema for incoming customer questions."""
    message: str = Field(..., min_length=1, max_length=5000, description="Customer question text")
    user_id: str = Field(..., min_length=1, max_length=100, description="Unique user or customer identifier")
    session_id: str | None = Field(default=None, max_length=100, description="Optional session ID for memory retention")


class AgentResponse(BaseModel):
    """Schema for outgoing assistant responses."""
    user_message: str = Field(..., description="The original prompt sent by the user")
    agent_reply: str = Field(..., description="The finalized text response from the ADK Agent")
    user_id: str = Field(..., description="The user identifier")
    session_id: str = Field(..., description="The active session identifier")


# ==============================================================================
# 6. REST API ENDPOINTS
# ==============================================================================

@app.get("/")
async def health_check():
    """Health check endpoint displaying agent metadata and status."""
    return {
        "status": "healthy",
        "agent": root_agent.name,
        "framework": "Google ADK (Agent Development Kit)",
        "docs_url": "/docs"
    }


@app.post("/chat", response_model=AgentResponse)
async def chat_with_agent(query: UserQuery):
    """
    Main endpoint for interacting with the Google ADK Agent.
    
    Flow:
    1. Resolve or generate session_id.
    2. Check or create session in InMemorySessionService.
    3. Run the ADK runner asynchronously.
    4. Automatically execute any required tools (e.g. track_order, cancel_order).
    5. Return the finalized conversational reply to the client.
    """
    # Auto-generate a session UUID if client did not supply one
    session_id = query.session_id or str(uuid4())

    try:
        # Retrieve or initialize the multi-turn session in ADK memory
        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=query.user_id,
            session_id=session_id,
        )

        if session is None:
            await session_service.create_session(
                app_name=APP_NAME,
                user_id=query.user_id,
                session_id=session_id,
            )

        # Wrap incoming text in Gemini types.Content
        user_msg = types.Content(
            role="user",
            parts=[types.Part(text=query.message)],
        )

        # Run the agent pipeline asynchronously
        final_reply = ""
        async for event in runner.run_async(
            user_id=query.user_id,
            session_id=session_id,
            new_message=user_msg,
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_reply = "".join(
                        part.text for part in event.content.parts if part.text
                    )

        if not final_reply:
            final_reply = "I apologize, but I could not generate a response. Please try again."

        return AgentResponse(
            user_message=query.message,
            agent_reply=final_reply,
            user_id=query.user_id,
            session_id=session_id,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Agent runtime error: {str(exc)}",
        ) from exc


# ==============================================================================
# 7. SERVER ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    # Start the development ASGI server on port 8000
    print("\n🚀 Starting Google ADK FastAPI server on http://127.0.0.1:8000")
    print("📖 Swagger Interactive Docs available at http://127.0.0.1:8000/docs\n")
    uvicorn.run("adk_fastapi_server:app", host="127.0.0.1", port=8000, reload=True)
