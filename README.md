# 🤖 AI Agents Masterclass: CrewAI & Google ADK Starter Kit

> **A production-ready, hands-on learning repository for building autonomous AI agents and multi-agent workflows using Google Gemini 3.8 Flash, CrewAI, and Google ADK (Agent Development Kit).**

---

## 📂 Repository Architecture

```text
├── crewAI_Agent.py             # Track 1: Multi-agent crew (Researcher + Writer)
├── CREWAI_BEGINNER_GUIDE.md    # Track 1: Full step-by-step CrewAI curriculum
│
├── adk_agent.py                # Track 2: Google ADK Autonomous Agent with custom tools
├── adk_terminal_demo.py        # Track 2: Interactive terminal chat REPL
├── adk_fastapi_server.py       # Track 2: Production REST API endpoint (/chat)
├── GOOGLE_ADK_1HOUR_GUIDE.md   # Track 2: 60-Minute Google ADK workshop guide & notes
│
├── .env.example                # Environment variables template
└── README.md                   # This project overview
```

---

## ⚡ Quickstart Setup

### 1. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install crewai google-adk google-genai python-dotenv fastapi uvicorn pydantic
```

### 3. Configure API Key
```bash
cp .env.example .env
# Edit .env and paste your Google Gemini API key:
# GOOGLE_API_KEY="AIzaSy..."
```

---

## 🚀 Track 1: CrewAI (Multi-Agent Collaboration)

CrewAI allows you to coordinate multiple autonomous agents working together like a human department.

- 📖 **Guide:** [CREWAI_BEGINNER_GUIDE.md](file:///Users/rameshwer/crewai/CREWAI_BEGINNER_GUIDE.md)
- 🐍 **Code:** [crewAI_Agent.py](file:///Users/rameshwer/crewai/crewAI_Agent.py)

### Run the CrewAI Demo:
```bash
python3 crewAI_Agent.py
```

---

## ⚡ Track 2: Google ADK (Autonomous Tool-Calling Agent)

Google ADK (Agent Development Kit) enables Gemini to autonomously invoke custom Python tools and maintain conversational session memory.

- 📖 **Guide:** [GOOGLE_ADK_1HOUR_GUIDE.md](file:///Users/rameshwer/crewai/GOOGLE_ADK_1HOUR_GUIDE.md)
- 🐍 **Standalone Agent:** [adk_agent.py](file:///Users/rameshwer/crewai/adk_agent.py)
- 💬 **Interactive Terminal:** [adk_terminal_demo.py](file:///Users/rameshwer/crewai/adk_terminal_demo.py)
- 🌐 **FastAPI REST API:** [adk_fastapi_server.py](file:///Users/rameshwer/crewai/adk_fastapi_server.py)

### 1. Run the Standalone ADK Test:
```bash
python3 adk_agent.py
```

### 2. Chat Live in the Terminal:
```bash
python3 adk_terminal_demo.py
```

### 3. Serve via FastAPI (REST API):
```bash
python3 adk_fastapi_server.py
# Open Swagger documentation at: http://127.0.0.1:8000/docs
```

---

## 🎓 Core Concepts Comparison

| Framework | Best For | Core Primitives | State Handling |
| :--- | :--- | :--- | :--- |
| **CrewAI** | Multi-agent collaborative workflows (Researcher + Writer + QA) | `Agent`, `Task`, `Crew`, `Process` | Pipeline context passed sequentially between tasks |
| **Google ADK** | Autonomous tool calling, reactive agents, REST API serving | `Agent`, `Tools` (Python functions), `Runner`, `InMemorySessionService` | Native session memory keyed by `(user_id, session_id)` |
