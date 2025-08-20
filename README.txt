Hi there, I'm Ose


import streamlit as st
import re, io, time
from typing import List, Dict
from collections import defaultdict

# -------------------------------
# File reading
# -------------------------------
def read_file(uploaded_file) -> str:
    """Reads txt, pdf, or docx into plain text"""
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    elif name.endswith(".pdf"):
        from pypdf import PdfReader
        with io.BytesIO(data) as f:
            pdf = PdfReader(f)
            return "\n".join([p.extract_text() or "" for p in pdf.pages])
    elif name.endswith(".docx"):
        import docx
        doc = docx.Document(io.BytesIO(data))
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        st.warning("Unsupported file type")
        return ""

# -------------------------------
# Band detection
# -------------------------------
def find_band(text: str) -> str:
    m = re.search(r'\bBand\s*(\d{1,2})\b', text, flags=re.I)
    return f"Band {m.group(1)}" if m else ""

BAND_SUMMARIES = {
    "Band 2": "Entry-level clinical/administrative support.",
    "Band 3": "Experienced support worker/administrator.",
    "Band 4": "Senior assistant/associate.",
    "Band 5": "Newly qualified practitioner/analyst.",
    "Band 6": "Specialist/experienced practitioner.",
    "Band 7": "Advanced specialist/team lead.",
    "Band 8a": "Service/department leadership.",
    "Band 8b": "Senior management.",
    "Band 8c": "Head of service/associate director.",
    "Band 8d": "Director-level leadership.",
    "Band 9": "Executive-level leadership."
}

# -------------------------------
# Criteria extraction
# -------------------------------
CRITERIA_HEADERS = ["Essential","Desirable","Knowledge","Skills","Experience","Qualifications","Values","Behaviours"]

def extract_criteria(text: str) -> Dict[str, List[str]]:
    """Extract bullet-point criteria grouped under headers"""
    sections = defaultdict(list)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    current = None
    for l in lines:
        header_hit = None
        for h in CRITERIA_HEADERS:
            if re.match(rf'^{re.escape(h)}\b', l, flags=re.I):
                header_hit = h
                break
        if header_hit:
            current = header_hit
            continue
        if re.match(r'^[-*•–]\s+', l):
            if current is None:
                current = "Essential"
            cleaned = re.sub(r'^[-*•–]\s+', '', l)
            sections[current].append(cleaned)
    if not any(sections.values()):
        sections["Essential"] = lines
    return dict(sections)

# -------------------------------
# Sentence splitting
# -------------------------------
def split_sentences(text: str) -> List[str]:
    return [x.strip() for x in re.split(r'(?<=[\.\?\!])\s+(?=[A-Z0-9])', text) if x.strip()]

# -------------------------------
# Evidence selection
# -------------------------------
def select_supporting_sentences(cv_text: str, criterion: str, top_k: int=3) -> List[str]:
    crit_tokens = set(re.findall(r'[A-Za-z]{3,}', criterion.lower()))
    sents = split_sentences(cv_text)
    scored = []
    for s in sents:
        stoks = set(re.findall(r'[A-Za-z]{3,}', s.lower()))
        score = len(crit_tokens & stoks)
        scored.append((score, s))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [s for _, s in scored[:top_k] if _ > 0]

# -------------------------------
# STAR formatter
# -------------------------------
def star_format(evidence_sentences: List[str], criterion: str) -> str:
    if not evidence_sentences:
        return f"- {criterion}: I have transferable skills and am committed to developing further."
    situation = evidence_sentences[0]
    task = criterion
    action = evidence_sentences[1] if len(evidence_sentences) > 1 else situation
    result = ""
    for s in evidence_sentences:
        if any(w in s.lower() for w in ["improved","reduced","increased","achieved","delivered","successfully","impact"]):
            result = s
            break
    if not result and len(evidence_sentences) > 2:
        result = evidence_sentences[2]
    return f"- {criterion}\n  - Situation/Task: {situation}\n  - Action: {action}\n  - Result: {result or 'Positive outcome achieved'}"

# -------------------------------
# Criteria coverage map
# -------------------------------
def criteria_coverage(criteria: Dict[str,List[str]], cv_text: str) -> Dict[str,str]:
    coverage = {}
    for crit in criteria.get("Essential", []):
        matches = select_supporting_sentences(cv_text, crit, top_k=2)
        if matches:
            if any(any(w in s.lower() for w in ["improved","audit","led","trained","quality","safeguard"]) for s in matches):
                coverage[crit] = "✅ Strong"
            else:
                coverage[crit] = "⚠️ Partial"
        else:
            coverage[crit] = "❌ Missing"
    return coverage

# -------------------------------
# Rule-based SI generator
# -------------------------------
def build_rule_based(ps_text: str, cv_text: str, band: str, word_target: int) -> str:
    criteria = extract_criteria(ps_text)
    parts = [f"I am applying for this {band} role with a strong fit to NHS values."]
    for c in criteria.get("Essential", []):
        ev = select_supporting_sentences(cv_text, c, 3)
        parts.append(star_format(ev, c))
    parts.append(BAND_SUMMARIES.get(band,""))
    return "\n".join(parts)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="NHS SI Generator v2", page_icon="💙", layout="wide")
st.title("💙 NHS Supporting Information Generator (v2)")

with st.sidebar:
    band_choice = st.selectbox("Band", ["Auto-detect","Band 2","Band 3","Band 4","Band 5","Band 6","Band 7","Band 8a","Band 8b","Band 8c","Band 8d","Band 9"])
    word_target = st.slider("Target word count", 350, 1200, 700, 50)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Upload Person Spec")
    ps_file = st.file_uploader("PDF/DOCX/TXT", type=["pdf","docx","txt"], key="ps")
    ps_text = read_file(ps_file)
with col2:
    st.subheader("Upload Your CV")
    cv_file = st.file_uploader("PDF/DOCX/TXT", type=["pdf","docx","txt"], key="cv")
    cv_text = read_file(cv_file)

if st.button("Generate SI", type="primary"):
    if not ps_text or not cv_text:
        st.error("Upload both files")
        st.stop()

    band = band_choice if band_choice != "Auto-detect" else (find_band(ps_text) or "Band 5")
    st.write("### Band summary")
    st.write(BAND_SUMMARIES.get(band, "General NHS expectations"))

    result_text = build_rule_based(ps_text, cv_text, band, word_target)
    st.write("### Draft Supporting Information")
    st.write(result_text)

    # Character counter
    chars = len(result_text)
    st.info(f"Character count: {chars}")
    if chars > 3000:
        st.warning("⚠️ Over NHS Jobs character limit (3000).")
    elif chars < 1500:
        st.warning("⚠️ Likely too short (min ~1500).")

    # Coverage heatmap
    criteria = extract_criteria(ps_text)
    coverage = criteria_coverage(criteria, cv_text)
    st.write("### Essential Criteria Coverage")
    for crit, status in coverage.items():
        st.write(f"{status} – {crit}")

    # Download buttons
    ts = int(time.time())
    fname = f"supporting_info_{band.replace(' ','').lower()}_{ts}"
    st.download_button("Download as .txt", data=result_text.encode("utf-8"), file_name=f"{fname}.txt")

    try:
        from docx import Document
        doc = Document()
        doc.add_heading("Supporting Information", 0)
        for para in result_text.splitlines():
            doc.add_paragraph(para)
        buf = io.BytesIO()
        doc.save(buf)
        st.download_button("Download as .docx", data=buf.getvalue(), file_name=f"{fname}.docx")
    except Exception:
        st.caption("Install python-docx for Word export")


