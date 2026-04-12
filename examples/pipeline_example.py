"""
Example: Full multi-agent pipeline — Agent A → B → C
Patient data flows through 3 agents. None see it.

pip install codeastra
"""
from codeastra import BlindAgentMiddleware, CodeAstraClient

API_KEY = "sk-guard-xxx"

# ── Simulate 3 agents in a pipeline ──────────────────────────────────────────

class IntakeAgent:
    """Simulates an intake agent that reads patient data."""
    def run(self, input: str) -> dict:
        # In real life: hits your EHR, database, form submission
        return {
            "name":      "Maria Garcia",
            "email":     "maria.garcia@email.com",
            "ssn":       "987-65-4321",
            "dob":       "1990-07-04",
            "mrn":       "MRN-1122334",
            "diagnosis": "J06.9 Acute upper respiratory infection",
            "note":      "Patient needs follow-up in 2 weeks",
        }

class SchedulingAgent:
    """Simulates a scheduling agent that books appointments."""
    def run(self, input: str) -> dict:
        return {"scheduled": True, "slot": "Monday 10:00 AM", "patient_ref": input}

class BillingAgent:
    """Simulates a billing agent that sends invoices."""
    def run(self, input: str) -> dict:
        return {"invoice_sent": True, "amount": 150.00, "recipient": input}

# ── Wrap each agent ───────────────────────────────────────────────────────────

agent_a = BlindAgentMiddleware(
    IntakeAgent(),
    api_key=API_KEY,
    agent_id="intake-agent",
    pipeline_id="demo_pipeline_001",
    classification="phi",
    verbose=True,
)

agent_b = BlindAgentMiddleware(
    SchedulingAgent(),
    api_key=API_KEY,
    agent_id="scheduling-agent",
    pipeline_id="demo_pipeline_001",
    verbose=True,
)

agent_c = BlindAgentMiddleware(
    BillingAgent(),
    api_key=API_KEY,
    agent_id="billing-agent",
    pipeline_id="demo_pipeline_001",
    verbose=True,
)

# ── Run the pipeline ──────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Intake agent processes patient")
print("=" * 60)
intake_result = agent_a.run("Process new patient")
print(f"What agent_a returned to system: {intake_result}")
# {"name": "[CVT:NAME:A1B2]", "email": "[CVT:EMAIL:C3D4]", ...}
# Real values safe in vault

print("\n" + "=" * 60)
print("STEP 2: Intake grants tokens to scheduling agent")
print("=" * 60)
grant = agent_a.grant_to(
    "scheduling-agent",
    allowed_actions=["schedule_appointment"],
    purpose="Schedule 2-week follow-up",
)
print(f"Grant result: {grant['message']}")

print("\n" + "=" * 60)
print("STEP 3: Scheduling agent books appointment")
print("=" * 60)
name_token  = agent_a.tokens.get("name")
scheduling_result = agent_b.run(name_token)
print(f"What agent_b returned: {scheduling_result}")

print("\n" + "=" * 60)
print("STEP 4: Scheduling grants email token to billing")
print("=" * 60)
agent_b.grant_to(
    "billing-agent",
    allowed_actions=["send_invoice"],
)

print("\n" + "=" * 60)
print("STEP 5: Billing agent sends invoice")
print("=" * 60)
email_token = agent_a.tokens.get("email")
billing_result = agent_c.run(email_token)
print(f"What agent_c returned: {billing_result}")

print("\n" + "=" * 60)
print("AUDIT TRAIL — Full chain of custody")
print("=" * 60)
audit = agent_a.audit()
if audit:
    for entry in audit:
        print(f"  {entry.get('from_agent')} → {entry.get('to_agent')}: "
              f"{entry.get('token')} | action={entry.get('action_type')} | "
              f"authorized={entry.get('authorized')}")
else:
    print("  (Run with real API key to see live audit trail)")

print("\n✅ PROOF: Zero real data seen by any agent.")
print(f"   Tokens in pipeline: {list(agent_a.tokens.values())}")
