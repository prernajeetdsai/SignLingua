"""
app.py — SignLingua  |  streamlit run app.py
Requires Streamlit >= 1.33 for st.audio_input() (live microphone recording)
"""

import streamlit as st
import base64, io

from main import (
    image_pipeline,
    voice_translate_pipeline,
    voice_qa_pipeline,
    add_text_to_store,
    get_store_size,
    clear_vector_store,
    SUPPORTED_LANGUAGES,
    LANG_CODE_TO_NAME,
    RTL_LANGS,
)

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SignLingua",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,600;0,700;1,400&family=DM+Mono:wght@300;400;500&family=Outfit:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp { background: #09080a !important; color: #e4ddd3; }
.stApp { font-family: 'Outfit', sans-serif; }

/* noise */
.stApp::after {
    content:''; position:fixed; inset:0; pointer-events:none; z-index:9999;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
    opacity:.5;
}

#MainMenu, footer, header { visibility:hidden; }
.block-container { padding:0 2.2rem 4rem !important; max-width:1340px !important; }
[data-testid="stSidebar"] { background:#0d0c0e !important; border-right:1px solid #1e1b20; }
[data-testid="stSidebar"] .block-container { padding:2rem 1.5rem !important; }

/* ── Masthead ── */
.masthead {
    padding:3rem 0 2rem; border-bottom:1px solid #1e1b20;
    margin-bottom:0; display:flex; align-items:flex-end;
    justify-content:space-between; gap:1rem; flex-wrap:wrap;
}
.mh-title { font-family:'Cormorant Garamond',serif; font-size:4rem; font-weight:700;
    line-height:1; letter-spacing:-.025em; color:#f0e9df; }
.mh-title span { color:#c9a55e; }
.mh-sub { font-family:'DM Mono',monospace; font-size:.63rem; letter-spacing:.22em;
    text-transform:uppercase; color:#3e3845; margin-top:.5rem; }
.mh-meta { font-family:'DM Mono',monospace; font-size:.62rem; color:#2e2b32;
    letter-spacing:.08em; text-align:right; line-height:2; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background:transparent !important; border-bottom:1px solid #1e1b20 !important;
    gap:0 !important; padding:0 !important; margin-bottom:2.5rem;
}
.stTabs [data-baseweb="tab"] {
    font-family:'DM Mono',monospace !important; font-size:.66rem !important;
    letter-spacing:.18em !important; text-transform:uppercase !important;
    color:#3a3542 !important; padding:.95rem 2rem !important;
    border-radius:0 !important; background:transparent !important;
    border:none !important; position:relative; transition:color .2s;
}
.stTabs [data-baseweb="tab"]:hover { color:#c9a55e !important; }
.stTabs [aria-selected="true"] { color:#c9a55e !important; background:transparent !important; }
.stTabs [aria-selected="true"]::after {
    content:''; position:absolute; bottom:-1px; left:0; right:0;
    height:2px; background:#c9a55e;
}
[data-baseweb="tab-highlight"] { display:none !important; }

/* ── Section label ── */
.slbl { font-family:'DM Mono',monospace; font-size:.6rem; letter-spacing:.22em;
    text-transform:uppercase; color:#3e3845; margin-bottom:.8rem;
    padding-bottom:.45rem; border-bottom:1px solid #16141a; }

/* ── Pipeline bar ── */
.pipe-bar { display:flex; align-items:center; flex-wrap:wrap; margin-bottom:2.2rem;
    border:1px solid #16141a; border-radius:4px; overflow:hidden; }
.pipe-node { font-family:'DM Mono',monospace; font-size:.58rem; letter-spacing:.12em;
    text-transform:uppercase; color:#c9a55e; padding:.5rem .95rem;
    white-space:nowrap; background:#100e0c; border-right:1px solid #16141a; }
.pipe-node.grn { color:#5fb896; background:#0c100d; }
.pipe-node:last-child { border-right:none; }

/* ── File uploader ── */
[data-testid="stFileUploadDropzone"] {
    background:#0d0c0e !important; border:1px solid #1e1b20 !important;
    border-radius:4px !important; transition:border-color .2s !important; }
[data-testid="stFileUploadDropzone"]:hover { border-color:#c9a55e !important; }
[data-testid="stFileUploadDropzone"] * {
    color:#3e3845 !important; font-family:'DM Mono',monospace !important; font-size:.72rem !important; }

/* ── Mic recorder widget (st.audio_input) ── */
[data-testid="stAudioInput"] {
    background:#0d0c0e !important; border:1px solid #1e1b20 !important;
    border-radius:8px !important; padding:1rem !important; }
[data-testid="stAudioInput"] label {
    font-family:'DM Mono',monospace !important; font-size:.6rem !important;
    letter-spacing:.18em !important; text-transform:uppercase !important;
    color:#3e3845 !important; }
/* mic button glow */
[data-testid="stAudioInput"] button {
    border-radius:50% !important; transition:box-shadow .3s !important; }
[data-testid="stAudioInput"] button:hover {
    box-shadow:0 0 0 6px rgba(201,165,94,.18) !important; }

/* ── Selectbox ── */
.stSelectbox>label { font-family:'DM Mono',monospace !important; font-size:.6rem !important;
    letter-spacing:.18em !important; text-transform:uppercase !important; color:#3e3845 !important; }
.stSelectbox [data-baseweb="select"]>div {
    background:#0d0c0e !important; border:1px solid #1e1b20 !important;
    border-radius:4px !important; color:#e4ddd3 !important; font-family:'Outfit',sans-serif !important; }
.stSelectbox [data-baseweb="select"]>div:hover { border-color:#c9a55e !important; }

/* ── Text input ── */
.stTextInput>label { font-family:'DM Mono',monospace !important; font-size:.6rem !important;
    letter-spacing:.18em !important; text-transform:uppercase !important; color:#3e3845 !important; }
.stTextInput input { background:#0d0c0e !important; border:1px solid #1e1b20 !important;
    border-radius:4px !important; color:#e4ddd3 !important;
    font-family:'DM Mono',monospace !important; font-size:.76rem !important; padding:.6rem .9rem !important; }
.stTextInput input:focus { border-color:#c9a55e !important; box-shadow:none !important; }
.stTextInput input::placeholder { color:#3a3542 !important; }

/* ── Checkbox ── */
.stCheckbox>label { font-family:'DM Mono',monospace !important; font-size:.7rem !important; color:#5a5462 !important; }

/* ── Button ── */
.stButton>button {
    width:100%; background:#c9a55e !important; color:#09080a !important;
    border:none !important; border-radius:2px !important;
    font-family:'DM Mono',monospace !important; font-size:.66rem !important;
    letter-spacing:.2em !important; text-transform:uppercase !important;
    font-weight:500 !important; padding:.78rem 2rem !important;
    transition:all .2s !important; margin-top:.3rem; }
.stButton>button:hover {
    background:#dbbf7a !important; transform:translateY(-1px) !important;
    box-shadow:0 8px 24px rgba(201,165,94,.2) !important; }
/* ghost variant */
.ghost .stButton>button {
    background:transparent !important; border:1px solid #2a2730 !important;
    color:#5a5462 !important; box-shadow:none !important; }
.ghost .stButton>button:hover {
    border-color:#c9a55e !important; color:#c9a55e !important;
    transform:none !important; box-shadow:none !important; }

/* ── Result card ── */
.rcard { background:#0d0c0e; border:1px solid #18151d; border-radius:4px;
    padding:1.5rem 1.5rem 1.5rem 2rem; margin-bottom:1rem;
    position:relative; overflow:hidden; }
.rcard-bar { position:absolute; top:0; left:0; width:3px; height:100%;
    background:#c9a55e; border-radius:4px 0 0 4px; }
.rcard-bar.grn { background:#5fb896; }
.rcard-bar.blu { background:#5b9bd4; }
.rcard-step { font-family:'DM Mono',monospace; font-size:.56rem; letter-spacing:.24em;
    text-transform:uppercase; color:#c9a55e; margin-bottom:.65rem; }
.rcard-step.grn { color:#5fb896; }
.rcard-step.blu { color:#5b9bd4; }
.rcard-body { font-family:'Outfit',sans-serif; font-size:1rem; color:#cec7bd;
    line-height:1.75; font-weight:300; }
.rcard-body.rtl { direction:rtl; text-align:right; }

/* ── Badges ── */
.lbadge { display:inline-flex; align-items:center; gap:.45rem;
    background:#120f16; border:1px solid #1e1b24; border-radius:2px;
    padding:.28rem .85rem; font-family:'DM Mono',monospace; font-size:.62rem;
    color:#7a7282; letter-spacing:.1em; margin-bottom:1.3rem; }
.ldot { width:6px; height:6px; background:#c9a55e; border-radius:50%; flex-shrink:0; }
.ldot.grn { background:#5fb896; }

/* ── Audio ── */
.aud-lbl { font-family:'DM Mono',monospace; font-size:.56rem; letter-spacing:.24em;
    text-transform:uppercase; color:#c9a55e; margin-bottom:.55rem; }
.aud-lbl.grn { color:#5fb896; }
audio { width:100% !important; border-radius:4px !important; }
.dl-btn { display:inline-block; font-family:'DM Mono',monospace; font-size:.62rem;
    letter-spacing:.12em; text-transform:uppercase; color:#c9a55e !important;
    text-decoration:none !important; border:1px solid #c9a55e;
    padding:.42rem 1rem; border-radius:2px; transition:all .2s; margin-top:.25rem; }
.dl-btn:hover { background:#c9a55e; color:#09080a !important; }

/* ── Status strips ── */
.s-ok { background:#0b1a0f; border:1px solid #19391f; border-radius:4px;
    padding:.5rem 1rem; font-family:'DM Mono',monospace; font-size:.62rem;
    color:#4a9e62; letter-spacing:.1em; margin-bottom:1.1rem; }
.s-info { background:#0e0f1a; border:1px solid #1c1f40; border-radius:4px;
    padding:.5rem 1rem; font-family:'DM Mono',monospace; font-size:.62rem;
    color:#5b7fc4; letter-spacing:.1em; margin-bottom:.9rem; }

/* ── Empty state ── */
.empty-box { background:#0d0c0e; border:1px dashed #18151d; border-radius:4px;
    padding:4rem 2rem; text-align:center; }
.empty-icon { font-size:2rem; color:#24202a; margin-bottom:.7rem; }
.empty-txt { font-family:'Cormorant Garamond',serif; font-size:1.2rem;
    color:#28242e; font-style:italic; }

/* ── HR ── */
.hr { border:none; border-top:1px solid #16141a; margin:1.3rem 0; }

/* ── Tip ── */
.tip { background:#0d0c0e; border-left:2px solid #1e1b20;
    padding:.75rem 1.1rem; border-radius:0 4px 4px 0; margin-top:1rem; }
.tip p { font-family:'DM Mono',monospace; font-size:.6rem; color:#3a3542; line-height:1.8; }
.tip strong { color:#52485c; }

/* ── Chat ── */
.chat-u { display:flex; justify-content:flex-end; margin-bottom:.9rem; }
.chat-a { display:flex; justify-content:flex-start; margin-bottom:.9rem; }
.bub-u { background:#1c1724; border:1px solid #2a2430; border-radius:12px 12px 2px 12px;
    padding:.75rem 1.1rem; max-width:82%; font-family:'Outfit',sans-serif;
    font-size:.9rem; color:#cec7bd; line-height:1.65; }
.bub-a { background:#0f1a0f; border:1px solid #1a2e1c; border-radius:12px 12px 12px 2px;
    padding:.75rem 1.1rem; max-width:86%; font-family:'Outfit',sans-serif;
    font-size:.9rem; color:#c0d8c0; line-height:1.65; }
.bub-role { font-family:'DM Mono',monospace; font-size:.52rem; letter-spacing:.18em;
    text-transform:uppercase; margin-bottom:.35rem; }
.bub-role.u { color:#6a6070; }
.bub-role.a { color:#5fb896; }

/* ── RAG status ── */
.rag-st { background:#120f16; border:1px solid #1e1b20; border-radius:4px;
    padding:.45rem .9rem; font-family:'DM Mono',monospace; font-size:.6rem;
    color:#4a4452; letter-spacing:.1em; margin-bottom:1.1rem;
    display:flex; align-items:center; gap:.55rem; }
.rag-dot { width:7px; height:7px; border-radius:50%; background:#28242e; flex-shrink:0; }
.rag-dot.on { background:#5fb896; box-shadow:0 0 7px rgba(95,184,150,.45); }

/* ── Mic area label ── */
.mic-label { font-family:'DM Mono',monospace; font-size:.6rem; letter-spacing:.2em;
    text-transform:uppercase; color:#5a5462; margin-bottom:.5rem; }
.mic-hint { font-family:'DM Mono',monospace; font-size:.58rem; color:#3a3542;
    letter-spacing:.06em; margin-top:.45rem; line-height:1.7; }

/* scrollbar */
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:#09080a; }
::-webkit-scrollbar-thumb { background:#1e1b20; border-radius:2px; }
::-webkit-scrollbar-thumb:hover { background:#c9a55e; }

/* image preview */
[data-testid="stImage"] img { border-radius:4px; border:1px solid #1e1b20; }

/* spinner */
[data-testid="stSpinner"] p {
    font-family:'DM Mono',monospace !important; font-size:.7rem !important;
    color:#3e3845 !important; letter-spacing:.1em !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def dl_html(audio_bytes: bytes, name: str = "audio.mp3") -> str:
    b64 = base64.b64encode(audio_bytes).decode()
    return (f'<a class="dl-btn" href="data:audio/mp3;base64,{b64}" '
            f'download="{name}">⬇ &nbsp;Download MP3</a>')

def pipe_bar(steps: list, green: bool = False):
    cls = "grn" if green else ""
    nodes = "".join(f'<div class="pipe-node {cls}">{s}</div>' for s in steps)
    st.markdown(f'<div class="pipe-bar">{nodes}</div>', unsafe_allow_html=True)

def rcard(step: str, body: str, color: str = "amber", rtl: bool = False):
    bar = {"amber": "", "green": " grn", "blue": " blu"}.get(color, "")
    sc  = {"amber": "", "green": " grn", "blue": " blu"}.get(color, "")
    rc  = " rtl" if rtl else ""
    st.markdown(f"""<div class="rcard">
        <div class="rcard-bar{bar}"></div>
        <div class="rcard-step{sc}">{step}</div>
        <div class="rcard-body{rc}">{body.replace(chr(10),'<br>')}</div>
    </div>""", unsafe_allow_html=True)

def lbadge(txt: str, green: bool = False):
    dc = "grn" if green else ""
    st.markdown(f'<div class="lbadge"><span class="ldot {dc}"></span>{txt}</div>',
                unsafe_allow_html=True)

def slbl(txt: str):
    st.markdown(f'<div class="slbl">{txt}</div>', unsafe_allow_html=True)

def hr():
    st.markdown('<hr class="hr">', unsafe_allow_html=True)

def ok(msg: str):
    st.markdown(f'<div class="s-ok">✓ &nbsp;{msg}</div>', unsafe_allow_html=True)

def info(msg: str):
    st.markdown(f'<div class="s-info">◈ &nbsp;{msg}</div>', unsafe_allow_html=True)

def empty():
    st.markdown("""<div class="empty-box">
        <div class="empty-icon">◈</div>
        <div class="empty-txt">Results will appear here</div>
    </div>""", unsafe_allow_html=True)

def audio_out(audio_bytes: bytes, fname: str, green: bool = False):
    lc = "grn" if green else ""
    st.markdown(f'<div class="aud-lbl {lc}">Synthesized Speech</div>', unsafe_allow_html=True)
    st.audio(audio_bytes, format="audio/mp3")
    st.markdown(dl_html(audio_bytes, fname), unsafe_allow_html=True)

def mic_section(label: str, key: str, hint: str = "") -> object:
    """Render mic recorder with styled label. Returns audio_input value."""
    st.markdown(f'<div class="mic-label">{label}</div>', unsafe_allow_html=True)
    val = st.audio_input("", key=key, label_visibility="collapsed")
    if hint:
        st.markdown(f'<div class="mic-hint">{hint}</div>', unsafe_allow_html=True)
    return val


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="font-family:'Cormorant Garamond',serif;font-size:1.75rem;
        font-weight:700;color:#f0e9df;letter-spacing:-.01em;margin-bottom:.1rem;">
        Sign<span style="color:#c9a55e">Lingua</span>
    </div>
    <div style="font-family:'DM Mono',monospace;font-size:.56rem;letter-spacing:.2em;
        text-transform:uppercase;color:#3a3542;margin-bottom:1.8rem;">
        Configuration
    </div>""", unsafe_allow_html=True)

    slbl("Gemini API Key")
    gemini_key = st.text_input("key", type="password", placeholder="AIza...",
                               label_visibility="collapsed", key="gkey")
    st.markdown("""<div class="tip"><p>
        Required for <strong>Tab 3</strong> Voice Q&A.<br>
        Get yours at <strong>aistudio.google.com</strong>
    </p></div>""", unsafe_allow_html=True)

    hr()

    slbl("Knowledge Base (RAG)")
    n = get_store_size()
    dc = "on" if n > 0 else ""
    st.markdown(f'<div class="rag-st"><div class="rag-dot {dc}"></div>'
                f'Vector store &nbsp;·&nbsp; {n} chunks indexed</div>',
                unsafe_allow_html=True)

    kb_doc = st.file_uploader("Add .txt or .md document", type=["txt", "md"],
                               key="kb_doc", label_visibility="collapsed")
    if kb_doc and st.button("Index Document", key="idx_doc"):
        with st.spinner("Embedding…"):
            try:
                added = add_text_to_store(kb_doc.read().decode("utf-8", errors="replace"))
                st.success(f"Indexed {added} chunks.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    kb_txt = st.text_area("Or paste text to index", height=80,
                           placeholder="Paste any text…",
                           label_visibility="collapsed", key="kb_txt")
    if kb_txt and st.button("Index Text", key="idx_txt"):
        with st.spinner("Embedding…"):
            try:
                added = add_text_to_store(kb_txt)
                st.success(f"Indexed {added} chunks.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if n > 0:
        st.markdown('<div class="ghost">', unsafe_allow_html=True)
        if st.button("Clear Knowledge Base", key="clr_kb"):
            clear_vector_store()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    hr()
    st.markdown("""<div class="tip"><p>
        OCR text from <strong>Tab 1</strong> is auto-indexed so you can ask
        questions about sign boards in Tab 3.
    </p></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MASTHEAD
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="masthead">
  <div>
    <div class="mh-title">Sign<span>Lingua</span></div>
    <div class="mh-sub">OCR · Live Voice Translation · Gemini 2.5 Flash AI · FAISS Embeddings</div>
  </div>
  <div class="mh-meta">
    40+ Languages &nbsp;·&nbsp; Tesseract OCR<br>
    Gemini 2.5 Flash &nbsp;·&nbsp; FAISS RAG<br>
    gTTS &nbsp;·&nbsp; Live Mic Input
  </div>
</div>
<br>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "◈  Image → OCR → Speech",
    "◈  Live Voice → Translate → Speech",
    "◈  Live Voice Q&A  ·  Gemini AI",
])


# ╔═══════════════════════════════════════════════════════╗
# ║  TAB 1 — IMAGE OCR PIPELINE                          ║
# ╚═══════════════════════════════════════════════════════╝
with tab1:
    pipe_bar(["Upload Image", "OCR Extract", "Detect Language",
              "Translate", "Synthesize Speech", "Play / Download"])

    L, R = st.columns([5, 7], gap="medium")

    with L:
        slbl("01 — Upload Sign Board Image")
        img_file = st.file_uploader("drop", type=["jpg","jpeg","png","bmp","tiff","webp"],
                                    key="img_up", label_visibility="collapsed")
        if img_file:
            st.image(img_file, use_container_width=True)
            st.markdown(
                f'<p style="font-family:\'DM Mono\',monospace;font-size:.58rem;'
                f'color:#2e2b32;letter-spacing:.1em;margin-top:.3rem;">'
                f'{img_file.name} &nbsp;·&nbsp; {img_file.size/1024:.1f} KB</p>',
                unsafe_allow_html=True)

        hr()
        slbl("02 — Target Language")
        tl1 = st.selectbox("l", list(SUPPORTED_LANGUAGES.keys()), index=0,
                           key="tl1", label_visibility="collapsed")
        tc1 = SUPPORTED_LANGUAGES[tl1]

        idx_chk = st.checkbox("Auto-index OCR text for Q&A (Tab 3)", value=True, key="idxchk")
        hr()
        run1 = st.button("Execute Pipeline", key="run1")

        st.markdown("""<div class="tip"><p>
            <strong>Tips:</strong> High-contrast images give better OCR accuracy.<br>
            Enable the checkbox above to ask questions about this sign in Tab 3.
        </p></div>""", unsafe_allow_html=True)

    with R:
        slbl("Pipeline Output")
        if run1:
            if not img_file:
                st.error("Please upload an image.")
            else:
                with st.spinner("Running OCR pipeline…"):
                    try:
                        res = image_pipeline(img_file.read(), tc1, add_to_store=idx_chk)
                        ok("Pipeline completed")
                        lbadge(f"Detected: {res['source_language_name']}")
                        rcard("Step 01 · OCR Extraction", res["extracted_text"])
                        rcard(f"Step 02 · Translation → {tl1}", res["translated_text"],
                              rtl=(tc1 in RTL_LANGS))
                        audio_out(res["audio_bytes"], f"signlingua_{tc1}.mp3")
                        if idx_chk and res.get("chunks_indexed", 0):
                            info(f'{res["chunks_indexed"]} chunks indexed — ask about this sign in Tab 3.')
                    except ValueError as e:
                        st.error(str(e))
                    except RuntimeError as e:
                        st.error(str(e))
                        if "OCR" in str(e):
                            st.info("Install Tesseract: `sudo apt-get install tesseract-ocr`")
                    except Exception as e:
                        st.error(f"Unexpected: {e}")
        else:
            empty()


# ╔═══════════════════════════════════════════════════════╗
# ║  TAB 2 — LIVE VOICE TRANSLATE                        ║
# ╚═══════════════════════════════════════════════════════╝
with tab2:
    pipe_bar(["🎙 Record Voice", "Speech-to-Text", "Detect Language",
              "Translate", "Synthesize Speech", "Play / Download"], green=True)

    L2, R2 = st.columns([5, 7], gap="medium")

    with L2:
        # Live mic recorder
        voice_rec2 = mic_section(
            label="01 — Record Your Voice",
            key="mic_tab2",
            hint=(
                "Click the microphone button to start recording.<br>"
                "Click again (or the stop button) to finish.<br>"
                "Your audio is processed locally — never stored."
            )
        )

        hr()
        slbl("02 — Target Language")
        tl2 = st.selectbox("l2", list(SUPPORTED_LANGUAGES.keys()), index=0,
                           key="tl2", label_visibility="collapsed")
        tc2 = SUPPORTED_LANGUAGES[tl2]
        hr()
        run2 = st.button("Translate Recording", key="run2")

        st.markdown("""<div class="tip"><p>
            <strong>Tips:</strong> Speak clearly, face the mic, minimise background noise.<br>
            Uses Google Web Speech API — internet connection required.<br>
            Supports all 40+ languages; auto-detects the spoken language.
        </p></div>""", unsafe_allow_html=True)

    with R2:
        slbl("Pipeline Output")
        if run2:
            if not voice_rec2:
                st.error("Please record your voice first using the microphone above.")
            else:
                with st.spinner("Processing voice translation…"):
                    try:
                        audio_bytes = voice_rec2.read()
                        res = voice_translate_pipeline(audio_bytes, tc2)
                        ok("Translation completed")
                        lbadge(f"Detected: {res['detected_language_name']}", green=True)
                        rcard("Step 01 · Transcription", res["transcribed_text"])
                        rcard(f"Step 02 · Translation → {tl2}", res["translated_text"],
                              color="green", rtl=(tc2 in RTL_LANGS))
                        audio_out(res["audio_bytes"], f"translated_{tc2}.mp3", green=True)
                    except RuntimeError as e:
                        st.error(str(e))
                        if "speech" in str(e).lower():
                            st.info("Ensure you have an internet connection for speech recognition.")
                    except Exception as e:
                        st.error(f"Unexpected: {e}")
        else:
            empty()


# ╔═══════════════════════════════════════════════════════╗
# ║  TAB 3 — GEMINI LIVE VOICE Q&A                       ║
# ╚═══════════════════════════════════════════════════════╝
with tab3:
    pipe_bar(["🎙 Record Question", "Speech-to-Text", "RAG Retrieval",
              "Gemini 2.5 Flash", "Synthesize Answer", "Play / Download"], green=True)

    if "chat" not in st.session_state:
        st.session_state.chat = []

    L3, R3 = st.columns([5, 7], gap="medium")

    with L3:
        # Live mic recorder
        voice_rec3 = mic_section(
            label="01 — Ask Your Question (Live Voice)",
            key="mic_tab3",
            hint=(
                "Press the mic to record your question.<br>"
                "Press again to stop. Then click <strong>Ask Gemini</strong>.<br>"
                "Supports multi-turn conversation memory."
            )
        )

        hr()
        slbl("02 — Response Language")
        rl3 = st.selectbox("rl", list(SUPPORTED_LANGUAGES.keys()), index=0,
                           key="rl3", label_visibility="collapsed")
        rc3 = SUPPORTED_LANGUAGES[rl3]
        hr()

        n3 = get_store_size()
        dc3 = "on" if n3 > 0 else ""
        st.markdown(f'<div class="rag-st"><div class="rag-dot {dc3}"></div>'
                    f'Knowledge base &nbsp;·&nbsp; {n3} vectors loaded</div>',
                    unsafe_allow_html=True)

        run3 = st.button("Ask Gemini", key="run3")

        st.markdown('<div class="ghost">', unsafe_allow_html=True)
        if st.button("Clear Conversation", key="clr_chat"):
            st.session_state.chat = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""<div class="tip"><p>
            <strong>Gemini 2.5 Flash</strong> uses:<br>
            · Retrieved context from indexed signs &amp; docs (RAG)<br>
            · Full conversation history (multi-turn)<br>
            · General world knowledge<br><br>
            Add documents via the <strong>sidebar</strong> to ground answers.
        </p></div>""", unsafe_allow_html=True)

    with R3:
        slbl("Conversation")

        if not gemini_key:
            st.markdown('<div class="s-info">◈ &nbsp;Enter your Gemini API key in the sidebar to enable Voice Q&A.</div>',
                        unsafe_allow_html=True)

        # Render existing conversation
        for turn in st.session_state.chat:
            if turn["role"] == "user":
                st.markdown(f"""<div class="chat-u">
                  <div class="bub-u">
                    <div class="bub-role u">You</div>
                    {turn["content"]}
                  </div></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="chat-a">
                  <div class="bub-a">
                    <div class="bub-role a">◈ Gemini</div>
                    {turn["content"].replace(chr(10),'<br>')}
                  </div></div>""", unsafe_allow_html=True)

        # Process new question
        if run3:
            if not gemini_key:
                st.error("Add your Gemini API key in the sidebar.")
            elif not voice_rec3:
                st.error("Please record your question using the microphone above.")
            else:
                with st.spinner("Asking Gemini…"):
                    try:
                        audio_bytes = voice_rec3.read()
                        res = voice_qa_pipeline(
                            audio_bytes=audio_bytes,
                            api_key=gemini_key,
                            response_lang=rc3,
                            history=st.session_state.chat,
                        )
                        ok("Answer ready")
                        lbadge(f"Question language: {res['detected_language_name']}", green=True)

                        rcard("Your Question", res["transcribed_question"], color="blue")

                        if res.get("context_used"):
                            info("RAG: Relevant context retrieved from knowledge base.")

                        rcard(f"Gemini Answer ({rl3})", res["answer_text"],
                              color="green", rtl=(rc3 in RTL_LANGS))
                        audio_out(res["answer_audio_bytes"],
                                  f"gemini_answer_{rc3}.mp3", green=True)

                        # Update history (keep last 20 turns)
                        st.session_state.chat.append(
                            {"role": "user", "content": res["transcribed_question"]})
                        st.session_state.chat.append(
                            {"role": "assistant", "content": res["answer_text"]})
                        if len(st.session_state.chat) > 20:
                            st.session_state.chat = st.session_state.chat[-20:]

                        st.rerun()

                    except RuntimeError as e:
                        st.error(str(e))
                        if "Gemini" in str(e) or "API" in str(e):
                            st.info("Check your Gemini API key and Flash 2.5 access.")
                        elif "speech" in str(e).lower():
                            st.info("Speech recognition needs an internet connection.")
                    except Exception as e:
                        st.error(f"Unexpected: {e}")

        elif not st.session_state.chat:
            empty()


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style="border-top:1px solid #16141a;margin-top:3rem;padding:1.5rem 0 0;">
<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:.8rem;">
  <div style="font-family:'DM Mono',monospace;font-size:.57rem;color:#24202a;letter-spacing:.1em;">
    SignLingua &nbsp;·&nbsp; OCR · Live Voice Translation · Gemini AI
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:.57rem;color:#24202a;letter-spacing:.07em;">
    Tesseract · deep-translator · gTTS · SpeechRecognition · sentence-transformers · FAISS · Gemini 2.5 Flash
  </div>
</div></div>""", unsafe_allow_html=True)
