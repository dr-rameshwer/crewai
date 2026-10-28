# ==============================================================================
# IMPORTS
# ==============================================================================
import os                           # Reads environment variables (API keys) from the operating system
from dotenv import load_dotenv      # Loads key-value pairs from a local .env file into os.environ

# CrewAI core building blocks:
# Agent: Autonomous persona with a defined role, goal, and backstory
# Task: Specific assignment with explicit description and expected output format
# Crew: The orchestrator managing the collection of agents and tasks
# Process: Execution mode (sequential executes tasks one after another)
# LLM: CrewAI's model wrapper supporting Gemini, OpenAI, and local models
from crewai import Agent, Task, Crew, Process, LLM


# ==============================================================================
# 1. ENVIRONMENT & MODEL SETUP
# ==============================================================================
load_dotenv()  # Searches for .env and loads GOOGLE_API_KEY into environment

# Initialize the Gemini model for CrewAI
# model="gemini/gemini-3.8-flash": Uses provider/model_name naming convention
# api_key: Read from the environment variable GOOGLE_API_KEY
# temperature=0.3: Balances creative phrasing with factual precision
llm = LLM(
    model="gemini/gemini-3.8-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)


# ==============================================================================
# 2. DEFINING THE AGENTS (The Team Members)
# ==============================================================================

# AGENT 1: The Researcher
# Specializes in extracting objective facts, benchmarks, and technical data
researcher = Agent(
    role="Senior Technology Researcher",                                    # Job title defining authority domain
    goal="Discover the top 3 verified technical facts about {topic}",       # Measurable target with {topic} placeholder
    backstory=(                                                            # Perspective guiding tone and skepticism
        "You are a veteran software analyst who loves data and hates "
        "marketing hype. You only provide concrete facts and benchmarks."
    ),
    llm=llm,                                                                # The language model powering this agent
    verbose=True                                                            # Logs thoughts and actions to terminal in real time
)

# AGENT 2: The Writer
# Specializes in translating raw bullet points into polished, reader-friendly prose
writer = Agent(
    role="Technical Content Writer",                                        # Job title
    goal="Turn raw technical facts into a simple, beginner-friendly 2-paragraph post",
    backstory=(                                                            # Personality: empathetic developer educator
        "You are a friendly developer educator who explains complex "
        "tech topics so simply that anyone can understand."
    ),
    llm=llm,
    verbose=True
)


# ==============================================================================
# 3. DEFINING INTERCONNECTED TASKS
# ==============================================================================

# TASK 1: Deep Research
task_research = Task(
    description="Research the topic: '{topic}'. Identify the top 3 concrete technical reasons or advantages.",
    expected_output="A bulleted list of 3 verified technical facts with brief explanations.",
    agent=researcher                                                        # Assigned worker
)

# TASK 2: Writing & Synthesis
# Notice: CrewAI automatically passes the output of Task 1 as context into Task 2!
task_write = Task(
    description="Using the bulleted facts provided by the researcher, write a crisp 2-paragraph beginner summary on '{topic}'.",
    expected_output="A clean 2-paragraph Markdown post with a title and bullet points.",
    agent=writer                                                            # Assigned worker
)


# ==============================================================================
# 4. ASSEMBLING THE CREW
# ==============================================================================
my_crew = Crew(
    agents=[researcher, writer],                                            # List of participating agents
    tasks=[task_research, task_write],                                      # Sequential list of tasks to execute
    process=Process.sequential,                                             # Run Task 1 first, then Task 2
    verbose=True                                                            # Prints real-time agent handoff logs
)


# ==============================================================================
# 5. EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    # The topic to interpolate into {topic} placeholders across agents and tasks
    topic_input = "Why developers are switching from Flask to FastAPI"
    print(f"\n🚀 Starting Crew with topic: '{topic_input}'\n")

    # kickoff() starts the execution pipeline and passes the input variables
    result = my_crew.kickoff(inputs={"topic": topic_input})

    # Print the finalized article produced by the writer
    print("\n" + "=" * 50)
    print("🎉 FINAL DELIVERABLE PRODUCED BY THE CREW:")
    print("=" * 50)
    print(result)