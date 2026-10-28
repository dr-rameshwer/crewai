# 🤖 CrewAI Quickstart & Learning Hub

Welcome to the **CrewAI Starter Kit**! This repository is designed as an educational, production-ready template for learning and building multi-agent AI systems with CrewAI and Google Gemini.

---

## 🚀 Quick Navigation

- 📖 **[Comprehensive Step-by-Step Guide](file:///Users/rameshwer/crewai/CREWAI_BEGINNER_GUIDE.md)**: Full tutorial covering mental models, agent design, task dependencies, execution flow, common pitfalls, and practice exercises.
- 🐍 **[crewAI_Agent.py](file:///Users/rameshwer/crewai/crewAI_Agent.py)**: The complete, annotated Python script running a 2-agent sequential crew.
- 🔑 **[.env.example](file:///Users/rameshwer/crewai/.env.example)**: Environment variables template for API keys.

---

## ⚡ 30-Second Quickstart

```bash
# 1. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install crewai python-dotenv google-genai

# 3. Configure API key
cp .env.example .env
# Edit .env and paste your GOOGLE_API_KEY

# 4. Run the Crew!
python3 crewAI_Agent.py
```
