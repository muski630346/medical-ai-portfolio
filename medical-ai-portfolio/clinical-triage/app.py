import os
import json
import streamlit as st
from groq import Groq

# ─── CONFIG ───────────────────────────────────────────────────
os.environ["GROQ_API_KEY"] = "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"
client = Groq()

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

DEPARTMENTS = {
    "Pneumonia": "Pulmonology",
    "Pneumothorax": "Emergency/Thoracic Surgery",
    "Mass": "Oncology",
    "Nodule": "Pulmonology/Oncology",
    "Effusion": "Pulmonology/Cardiology",
    "Atelectasis": "Pulmonology",
    "Consolidation": "Pulmonology",
    "Edema": "Cardiology",
    "Cardiomegaly": "Cardiology",
    "Infiltration": "Pulmonology",
    "Emphysema": "Pulmonology",
    "Fibrosis": "Pulmonology",
    "Pleural_Thickening": "Pulmonology",
    "Hernia": "General Surgery"
}

SEVERITY_COLORS = {
    "P0": "#ef4444",
    "P1": "#f97316",
    "P2": "#eab308"
}

SEVERITY_LABELS = {
    "P0": "CRITICAL — Immediate Attention Required",
    "P1": "HIGH — Urgent Review Needed",
    "P2": "MEDIUM — Schedule Soon"
}

MODEL = "llama-3.3-70b-versatile"

# ─── AGENTS ───────────────────────────────────────────────────
def triage_agent(complaint: str, vitals: str) -> dict:
    prompt = f"""You are a clinical triage AI agent in a hospital.
Analyze this patient complaint and vitals carefully.

Patient Complaint: {complaint}
Vitals: {vitals}

Extract and return ONLY a JSON with these exact fields:
{{
  "symptoms": ["list", "of", "symptoms"],
  "duration": "how long symptoms present",
  "severity_indicators": ["any", "red", "flags"],
  "suspected_conditions": ["possible", "conditions"],
  "urgency_reasoning": "why this is urgent or not"
}}

Return ONLY valid JSON, no extra text."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500
    )
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"symptoms": [], "suspected_conditions": [],
                "urgency_reasoning": response.choices[0].message.content,
                "severity_indicators": []}

def risk_agent(triage_data: dict, patient_history: str) -> dict:
    prompt = f"""You are a medical risk assessment AI agent.

Triage Analysis: {triage_data}
Patient History: {patient_history}

Based on this information, determine:
1. Severity level (P0=Critical/life threatening, P1=Urgent, P2=Semi-urgent)
2. Key risk factors
3. Immediate actions needed

Return ONLY valid JSON:
{{
  "severity": "P0 or P1 or P2",
  "risk_factors": ["list", "of", "risks"],
  "immediate_actions": ["action1", "action2"],
  "monitoring_needed": ["what", "to", "monitor"],
  "risk_score": 85
}}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500
    )
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"severity": "P1", "risk_factors": [],
                "immediate_actions": [], "risk_score": 50,
                "monitoring_needed": []}

def routing_agent(triage_data: dict, risk_data: dict) -> dict:
    prompt = f"""You are a hospital routing AI agent.

Triage Data: {triage_data}
Risk Assessment: {risk_data}

Determine the best hospital routing for this patient.

Return ONLY valid JSON:
{{
  "primary_department": "department name",
  "secondary_department": "backup department",
  "specialist_needed": "type of specialist",
  "bed_priority": "ICU or General Ward or Outpatient",
  "estimated_wait": "immediate or 30 mins or 2 hours",
  "routing_reasoning": "why this routing"
}}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=400
    )
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"primary_department": "General Medicine",
                "secondary_department": "Emergency",
                "specialist_needed": "General Physician",
                "bed_priority": "General Ward",
                "estimated_wait": "30 mins",
                "routing_reasoning": "Default routing applied"}

def report_agent(complaint, vitals, history,
                 triage_data, risk_data, routing_data) -> str:
    prompt = f"""You are a medical report generation AI.

Create a professional clinical triage report based on:
- Complaint: {complaint}
- Vitals: {vitals}
- History: {history}
- Triage: {triage_data}
- Risk: {risk_data}
- Routing: {routing_data}

Write a concise clinical report in 150 words covering:
1. Patient presentation summary
2. Key findings and risk factors
3. Severity assessment with reasoning
4. Recommended immediate actions
5. Department routing with justification

Use professional medical language."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=600
    )
    return response.choices[0].message.content

# ─── STREAMLIT UI ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="ClinIQ — Clinical Triage AI",
        page_icon="🏥",
        layout="wide"
    )

    st.markdown("""
    <div style='background: linear-gradient(135deg, #0f172a, #1e3a5f);
                padding: 2rem; border-radius: 12px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin:0;'>🏥 ClinIQ — Clinical Triage AI System</h1>
        <p style='color: #94a3b8; margin:0;'>
            Multi-Agent AI for Patient Severity Classification & Hospital Routing
        </p>
        <p style='color: #64748b; font-size: 12px; margin-top: 8px;'>
            Built by Shaik Muskan | Vardhaman College of Engineering |
            Medical AI Portfolio Project
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div style='background:#ef444420; border:1px solid #ef4444;
            border-radius:8px; padding:10px; text-align:center;'>
            <b style='color:#ef4444;'>🔴 P0 — CRITICAL</b><br>
            <small>Immediate life-saving action</small></div>""",
            unsafe_allow_html=True)
    with col2:
        st.markdown("""<div style='background:#f9731620; border:1px solid #f97316;
            border-radius:8px; padding:10px; text-align:center;'>
            <b style='color:#f97316;'>🟠 P1 — URGENT</b><br>
            <small>Within 30 minutes</small></div>""",
            unsafe_allow_html=True)
    with col3:
        st.markdown("""<div style='background:#eab30820; border:1px solid #eab308;
            border-radius:8px; padding:10px; text-align:center;'>
            <b style='color:#eab308;'>🟡 P2 — SEMI-URGENT</b><br>
            <small>Within 2 hours</small></div>""",
            unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📋 Patient Information")

    col_left, col_right = st.columns(2)
    with col_left:
        complaint = st.text_area(
            "Chief Complaint *",
            placeholder="e.g. Severe chest pain, shortness of breath...",
            height=120
        )
        vitals = st.text_area(
            "Vital Signs *",
            placeholder="e.g. BP: 140/90, HR: 110 bpm, SpO2: 94%...",
            height=100
        )
    with col_right:
        patient_history = st.text_area(
            "Patient History",
            placeholder="e.g. 55 year old male, diabetic...",
            height=120
        )
        age    = st.number_input("Patient Age", 1, 120, 45)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    st.markdown("**Quick Test Examples:**")
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        if st.button("🔴 Critical Case"):
            st.session_state['complaint'] = \
                "Severe chest pain radiating to left arm, " \
                "difficulty breathing, sweating profusely for 1 hour"
            st.session_state['vitals'] = \
                "BP: 160/100, HR: 130 bpm, SpO2: 88%, Temp: 37.2°C"
            st.session_state['history'] = \
                "65 year old male, smoker, diabetic, " \
                "family history of heart disease"
            st.rerun()
    with ex2:
        if st.button("🟠 Urgent Case"):
            st.session_state['complaint'] = \
                "High fever for 3 days, productive cough with blood, " \
                "shortness of breath on exertion"
            st.session_state['vitals'] = \
                "BP: 120/80, HR: 95 bpm, Temp: 39.5°C, SpO2: 92%"
            st.session_state['history'] = \
                "35 year old female, no significant history"
            st.rerun()
    with ex3:
        if st.button("🟡 Semi-Urgent Case"):
            st.session_state['complaint'] = \
                "Persistent dry cough for 2 weeks, " \
                "mild breathlessness, no fever"
            st.session_state['vitals'] = \
                "BP: 118/76, HR: 78 bpm, Temp: 37.0°C, SpO2: 97%"
            st.session_state['history'] = \
                "42 year old female, non-smoker, mild asthma"
            st.rerun()

    if 'complaint' in st.session_state:
        complaint = st.session_state['complaint']
    if 'vitals' in st.session_state:
        vitals = st.session_state['vitals']
    if 'history' in st.session_state:
        patient_history = st.session_state['history']

    st.markdown("---")

    if st.button("🚀 Run Clinical Triage Analysis",
                 type="primary", use_container_width=True):
        if not complaint or not vitals:
            st.error("Please enter patient complaint and vitals!")
            return

        with st.spinner("Running 4-agent clinical analysis..."):
            progress = st.progress(0)
            status   = st.empty()

            status.markdown("🤖 **Agent 1:** Analyzing symptoms...")
            progress.progress(20)
            triage_data = triage_agent(complaint, vitals)

            status.markdown("🤖 **Agent 2:** Assessing risk level...")
            progress.progress(45)
            risk_data = risk_agent(triage_data, patient_history)

            status.markdown("🤖 **Agent 3:** Routing to department...")
            progress.progress(70)
            routing_data = routing_agent(triage_data, risk_data)

            status.markdown("🤖 **Agent 4:** Generating clinical report...")
            progress.progress(90)
            report = report_agent(
                complaint, vitals, patient_history,
                triage_data, risk_data, routing_data
            )
            progress.progress(100)
            status.empty()

        severity = risk_data.get("severity", "P1")
        color    = SEVERITY_COLORS.get(severity, "#eab308")
        label    = SEVERITY_LABELS.get(severity, "")

        st.markdown(f"""
        <div style='background:{color}22; border:2px solid {color};
                    border-radius:12px; padding:1.5rem;
                    text-align:center; margin: 1rem 0;'>
            <h2 style='color:{color}; margin:0;'>
                {severity} — {label}
            </h2>
            <p style='color:{color}; margin:4px 0 0;'>
                Risk Score: {risk_data.get("risk_score", "N/A")}/100
            </p>
        </div>
        """, unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown("### 🔍 Triage Analysis")
            st.markdown("**Symptoms:**")
            for s in triage_data.get("symptoms", []):
                st.markdown(f"• {s}")
            st.markdown("**Suspected Conditions:**")
            for c in triage_data.get("suspected_conditions", []):
                st.markdown(f"• {c}")
            flags = triage_data.get("severity_indicators", [])
            if flags:
                st.error("⚠️ Red Flags: " + ", ".join(flags))

        with r2:
            st.markdown("### ⚠️ Risk Assessment")
            st.markdown("**Risk Factors:**")
            for r in risk_data.get("risk_factors", []):
                st.markdown(f"• {r}")
            st.markdown("**Immediate Actions:**")
            for a in risk_data.get("immediate_actions", []):
                st.success(f"✓ {a}")
            st.markdown("**Monitor:**")
            for m in risk_data.get("monitoring_needed", []):
                st.markdown(f"• {m}")

        with r3:
            st.markdown("### 🏥 Hospital Routing")
            dept = routing_data.get("primary_department", "General Medicine")
            bed  = routing_data.get("bed_priority", "General Ward")
            wait = routing_data.get("estimated_wait", "30 mins")
            spec = routing_data.get("specialist_needed", "General Physician")
            st.markdown(f"""
            <div style='background:#0f172a; border-radius:10px; padding:1rem;'>
                <p style='color:#34d399; margin:6px 0;'>
                    <b>Primary:</b> {dept}</p>
                <p style='color:#60a5fa; margin:6px 0;'>
                    <b>Specialist:</b> {spec}</p>
                <p style='color:#f59e0b; margin:6px 0;'>
                    <b>Bed Priority:</b> {bed}</p>
                <p style='color:#a78bfa; margin:6px 0;'>
                    <b>Wait Time:</b> {wait}</p>
            </div>
            """, unsafe_allow_html=True)
            if routing_data.get("routing_reasoning"):
                st.info(routing_data["routing_reasoning"])

        st.markdown("---")
        st.markdown("### 📄 Clinical Report")
        st.markdown(f"""
        <div style='background:#0f172a; border:1px solid #334155;
                    border-radius:10px; padding:1.5rem;
                    color:#e2e8f0; line-height:1.8;'>
            {report}
        </div>
        """, unsafe_allow_html=True)

        full_report = f"""
CLINIQ CLINICAL TRIAGE REPORT
==============================
Patient Age: {age} | Gender: {gender}
Complaint: {complaint}
Vitals: {vitals}
History: {patient_history}

SEVERITY: {severity} — {label}
Risk Score: {risk_data.get('risk_score', 'N/A')}/100

TRIAGE ANALYSIS:
Symptoms: {', '.join(triage_data.get('symptoms', []))}
Suspected: {', '.join(triage_data.get('suspected_conditions', []))}

RISK FACTORS: {', '.join(risk_data.get('risk_factors', []))}
IMMEDIATE ACTIONS: {', '.join(risk_data.get('immediate_actions', []))}

ROUTING: {routing_data.get('primary_department')}
BED: {routing_data.get('bed_priority')}
WAIT: {routing_data.get('estimated_wait')}

CLINICAL REPORT:
{report}

Generated by ClinIQ AI | Shaik Muskan | Vardhaman College
For research purposes only.
        """
        st.download_button(
            "📥 Download Full Report",
            full_report,
            file_name="cliniq_triage_report.txt",
            use_container_width=True
        )
        st.warning(
            "⚕️ DISCLAIMER: ClinIQ is an AI research tool only. "
            "All clinical decisions must be made by qualified "
            "medical professionals."
        )

if __name__ == "__main__":
    main()