"""
Example: CrewAI multi-agent pipeline — all agents blind.

pip install codeastra crewai
"""
from crewai import Agent, Task, Crew
from crewai.tools import tool
from codeastra import BlindCrewAIAgent

# ── Your normal CrewAI tools ──────────────────────────────────────────────────

@tool("EHR Lookup")
def ehr_lookup(patient_id: str) -> dict:
    """Fetch patient record from EHR system."""
    return {
        "name":      "Alice Johnson",
        "email":     "alice@example.com",
        "dob":       "1975-03-22",
        "mrn":       "MRN-7362819",
        "diagnosis": "E11.9 Type 2 diabetes",
        "phone":     "+1-415-555-0199",
    }

@tool("Schedule Appointment")
def schedule_appointment(patient_token: str, slot: str) -> dict:
    """Schedule an appointment for a patient."""
    return {"scheduled": True, "slot": slot, "patient": patient_token}

@tool("Send Invoice")
def send_invoice(email_token: str, amount: float) -> dict:
    """Send invoice to patient email."""
    return {"sent": True, "to": email_token, "amount": amount}

# ── Normal CrewAI agents ──────────────────────────────────────────────────────

intake_agent = Agent(
    role="Patient Intake Specialist",
    goal="Collect and verify patient information",
    backstory="Expert at processing patient intake efficiently",
    tools=[ehr_lookup],
)

scheduling_agent = Agent(
    role="Appointment Scheduler",
    goal="Schedule patient appointments",
    backstory="Manages the clinic appointment calendar",
    tools=[schedule_appointment],
)

billing_agent = Agent(
    role="Billing Specialist",
    goal="Process patient billing",
    backstory="Handles patient invoicing and payments",
    tools=[send_invoice],
)

# ── Wrap each agent — two lines each ─────────────────────────────────────────

blind_intake = BlindCrewAIAgent(
    intake_agent,
    api_key="sk-guard-xxx",
    agent_id="intake",
    pipeline_id="patient_flow_001",
    classification="phi",
    verbose=True,
)

blind_scheduling = BlindCrewAIAgent(
    scheduling_agent,
    api_key="sk-guard-xxx",
    agent_id="scheduling",
    pipeline_id="patient_flow_001",
)

blind_billing = BlindCrewAIAgent(
    billing_agent,
    api_key="sk-guard-xxx",
    agent_id="billing",
    pipeline_id="patient_flow_001",
)

# ── Tasks ─────────────────────────────────────────────────────────────────────

intake_task = Task(
    description="Look up patient P001 and collect their information",
    agent=blind_intake,
    expected_output="Patient information collected and tokenized",
)

scheduling_task = Task(
    description="Schedule a follow-up appointment for the patient",
    agent=blind_scheduling,
    expected_output="Appointment scheduled",
)

billing_task = Task(
    description="Send an invoice for $150 to the patient",
    agent=blind_billing,
    expected_output="Invoice sent",
)

# ── Run the crew ──────────────────────────────────────────────────────────────

crew = Crew(
    agents=[blind_intake, blind_scheduling, blind_billing],
    tasks=[intake_task, scheduling_task, billing_task],
    verbose=True,
)

result = crew.kickoff()

# After intake runs, grant tokens to scheduling
blind_intake.grant_to("scheduling", allowed_actions=["schedule_appointment"])

# After scheduling runs, grant subset to billing
blind_scheduling.grant_to("billing", allowed_actions=["send_invoice"])

print("\n--- AUDIT TRAIL ---")
audit = blind_intake.audit()
for entry in audit:
    print(f"  {entry['from_agent']} → {entry['to_agent']}: {entry['token']} ({entry['action_type']})")
print("No agent ever saw real patient data.")
