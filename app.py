import streamlit as st
from pypdf import PdfReader
import re
from core.retrieval_pipeline import analyze_contract
st.set_page_config(page_title="Clauser AI", layout="wide")

@st.cache_resource
def load_pipeline():
    return analyze_contract


st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.header {
    text-align: center;
    font-size: 32px;
    font-weight: 600;
}
.subtle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 20px;
}
.section {
    margin-top: 20px;
    margin-bottom: 10px;
    font-size: 20px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'>Clauser AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtle'>Clause-level risk analysis using retrieval grounded in Indian Contract Law</div>", unsafe_allow_html=True)

st.warning(
    "This tool is for educational purposes only and does NOT constitute legal advice. "
    "Do not rely on this output for legal decisions."
)
st.markdown("<div class='section'>Input</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload contract (PDF or TXT)", type=["pdf", "txt"])

text = ""

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    else:
        text = uploaded_file.read().decode("utf-8")
    st.success("File loaded")
    with st.expander("View extracted text"):
        st.text(text[:3000])
else:
    text = st.text_area("Paste contract text", height=200, placeholder="Paste your contract here...")
agree = st.checkbox(
    "I understand this tool is for educational purposes only and not legal advice."
)
analyze_btn = st.button(
    "Analyze Contract",
    use_container_width=True,
    disabled=not agree
)

def clean_output(text):
    return re.sub(r'\n\s*\n+', '\n\n', text).strip()

def parse_clause(text):
    data = {"type": "", "risk": "", "why": [], "legal": []}
    lines = text.split("\n")
    current = None
    for line in lines:
        line = line.strip()
        if line.startswith("Clause Type"):
            data["type"] = line.split(":",1)[-1].strip()
        elif line.startswith("Risk Level"):
            data["risk"] = line.split(":",1)[-1].strip()
        elif "Why Risky" in line:
            current = "why"
        elif "Legal Basis" in line:
            current = "legal"
        elif line.startswith("-"):
            if current:
                data[current].append(line[1:].strip())
    return data

if analyze_btn:
    if not agree:
        st.warning("Please acknowledge the disclaimer before proceeding.")
        st.stop()

    if not text or len(text.strip()) < 50:
        st.warning("Please enter a valid contract (at least 50+ characters).")
        st.stop()

    with st.spinner("Analyzing contract..."):
        pipeline = load_pipeline()
        res = pipeline(text)

    if not res.get("results") or res["summary"].startswith("Input too short"):
        st.warning("Input is too short or not a valid contract.")
        st.stop()

    st.markdown("<div class='section'>Overview</div>", unsafe_allow_html=True)

    score = res["score"]
    level = res["risk_level"]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Risk Score", score)

    with c2:
        if "CRITICAL" in level:
            st.error(level)
        elif "HIGH" in level:
            st.warning(level)
        elif "MODERATE" in level:
            st.info(level)
        else:
            st.success(level)

    with c3:
        st.info(res["summary"])

    st.markdown("<div class='section'>Breakdown</div>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    b1.metric("High", res["risk_count"]["HIGH"])
    b2.metric("Medium", res["risk_count"]["MEDIUM"])
    b3.metric("Low", res["risk_count"]["LOW"])

    st.caption(f"{len(res['results'])} clauses analyzed")

    st.markdown("<div class='section'>Clause Analysis</div>", unsafe_allow_html=True)

    for i, r in enumerate(res["results"], 1):
        parsed = parse_clause(clean_output(r))
        risk = parsed["risk"]

        with st.expander(f"Clause {i} — {parsed['type']} ({risk})"):
            if "HIGH" in risk:
                st.error(f"Risk Level: {risk}")
            elif "MEDIUM" in risk:
                st.warning(f"Risk Level: {risk}")
            else:
                st.success(f"Risk Level: {risk}")

            st.markdown("**Why Risky:**")
            for w in parsed["why"]:
                st.markdown(f"- {w}")

            st.markdown("**Legal Basis:**")
            for l in parsed["legal"]:
                st.markdown(f"- {l}")

            st.markdown("---")