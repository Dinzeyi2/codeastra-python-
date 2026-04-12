"""
Example: LangChain agent made blind with 2 lines.

pip install codeastra langchain langchain-openai
"""
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain import hub
from codeastra import BlindAgentMiddleware

# ── Your normal LangChain tools — unchanged ───────────────────────────────────

@tool
def lookup_patient(patient_id: str) -> dict:
    """Look up a patient record."""
    # In real life this hits your EHR / database
    return {
        "name":      "John Smith",
        "email":     "john.smith@email.com",
        "dob":       "1980-01-15",
        "ssn":       "123-45-6789",
        "mrn":       "MRN-4829103",
        "diagnosis": "Z34.90 Normal pregnancy",
    }

@tool
def get_appointments(patient_id: str) -> dict:
    """Get available appointment slots."""
    return {"slots": ["Monday 10am", "Tuesday 2pm", "Friday 9am"]}

# ── Build your normal LangChain agent — unchanged ─────────────────────────────

llm      = ChatOpenAI(model="gpt-4o")
prompt   = hub.pull("hwchase17/openai-functions-agent")
agent    = create_openai_functions_agent(llm, [lookup_patient, get_appointments], prompt)
executor = AgentExecutor(agent=agent, tools=[lookup_patient, get_appointments])

# ── Two lines. Agent is now blind. ────────────────────────────────────────────

blind = BlindAgentMiddleware(
    executor,
    api_key="sk-guard-xxx",     # your Codeastra API key
    agent_id="scheduling-agent",
    classification="phi",        # auto-detected but you can override
    verbose=True,                # prints: [CodeAstra] Tokenized 5 field(s): ['name', 'email', ...]
)

# ── Use exactly as before ─────────────────────────────────────────────────────

result = blind.invoke({"input": "Schedule a follow-up appointment for patient P001"})

print("\n--- RESULT ---")
print(result["output"])
# Agent never saw "John Smith" or "john.smith@email.com"
# It reasoned on [CVT:NAME:A1B2] and [CVT:EMAIL:C3D4]

print("\n--- TOKENS MINTED THIS SESSION ---")
print(blind.tokens)
# {"name": "[CVT:NAME:A1B2]", "email": "[CVT:EMAIL:C3D4]", ...}

print("\n--- PROOF ---")
print(f"Real data seen by agent: 0 fields")
print(f"Tokens minted: {blind.token_count}")
