"""
main.py — SignLingua Core Backend
──────────────────────────────────
STT  : OpenAI Whisper (runs locally, no API key, no internet, no rate limits)
       Replaces Google Web Speech API which fails on cloud servers.
OCR  : Tesseract
Trans: Google Translate via deep-translator
TTS  : gTTS
AI   : Gemini 2.5 Flash + FAISS RAG embeddings
"""

import os, io, tempfile, textwrap
from typing import Optional


# ════════════════════════════════════════════════════════════════
# SUPPORTED LANGUAGES
# ════════════════════════════════════════════════════════════════

SUPPORTED_LANGUAGES: dict[str, str] = {
    "English": "en", "Hindi": "hi", "French": "fr", "Spanish": "es",
    "German": "de", "Chinese (Simplified)": "zh-cn",
    "Chinese (Traditional)": "zh-tw", "Japanese": "ja", "Korean": "ko",
    "Arabic": "ar", "Portuguese": "pt", "Russian": "ru", "Italian": "it",
    "Dutch": "nl", "Turkish": "tr", "Bengali": "bn", "Tamil": "ta",
    "Telugu": "te", "Marathi": "mr", "Gujarati": "gu", "Kannada": "kn",
    "Malayalam": "ml", "Punjabi": "pa", "Urdu": "ur", "Thai": "th",
    "Vietnamese": "vi", "Indonesian": "id", "Malay": "ms", "Greek": "el",
    "Polish": "pl", "Swedish": "sv", "Norwegian": "no", "Danish": "da",
    "Finnish": "fi", "Czech": "cs", "Romanian": "ro", "Hungarian": "hu",
    "Ukrainian": "uk", "Hebrew": "iw", "Swahili": "sw", "Afrikaans": "af",
}

LANG_CODE_TO_NAME: dict[str, str] = {v: k for k, v in SUPPORTED_LANGUAGES.items()}
RTL_LANGS = {"ar", "he", "ur", "iw", "fa"}

# Whisper model cache (lazy-load once per session)
_whisper_model = None


# ════════════════════════════════════════════════════════════════
# OCR
# ════════════════════════════════════════════════════════════════

def extract_text_from_image(image_bytes: bytes) -> str:
    """Tesseract OCR with preprocessing for higher accuracy."""
    try:
        import pytesseract
        from PIL import Image, ImageEnhance, ImageFilter

        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        img = ImageEnhance.Contrast(img).enhance(1.8)
        img = ImageEnhance.Sharpness(img).enhance(2.2)
        img = img.filter(ImageFilter.MedianFilter(size=3))

        best = ""
        for psm in ("--psm 6", "--psm 11", "--psm 3", "--psm 4"):
            try:
                t = pytesseract.image_to_string(img, config=psm).strip()
                if len(t) > len(best):
                    best = t
            except Exception:
                continue

        if not best:
            raise ValueError("No text could be extracted from the image.")
        return best

    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"OCR failed: {exc}") from exc


# ════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ════════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    """Return ISO-639-1 code or 'unknown'."""
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42
        return detect(text)
    except Exception:
        return "unknown"


# ════════════════════════════════════════════════════════════════
# TRANSLATION
# ════════════════════════════════════════════════════════════════

def translate_text(text: str, target: str, source: str = "auto") -> dict:
    """Google Translate via deep-translator."""
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source=source, target=target).translate(text)
        src = detect_language(text) if source == "auto" else source
        return {"translated_text": translated, "source_language": src, "target_language": target}
    except Exception as exc:
        raise RuntimeError(f"Translation failed: {exc}") from exc


# ════════════════════════════════════════════════════════════════
# TEXT-TO-SPEECH
# ════════════════════════════════════════════════════════════════

def text_to_speech(text: str, lang_code: str) -> bytes:
    """gTTS → MP3 bytes."""
    try:
        from gtts import gTTS
        code_map = {"zh-cn": "zh-CN", "zh-tw": "zh-TW"}
        code = code_map.get(lang_code, lang_code)
        buf = io.BytesIO()
        gTTS(text=text, lang=code, slow=False).write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as exc:
        raise RuntimeError(f"TTS failed: {exc}") from exc


# ════════════════════════════════════════════════════════════════
# SPEECH-TO-TEXT  —  OpenAI Whisper (LOCAL, no API key needed)
# ════════════════════════════════════════════════════════════════

def _get_whisper_model(model_size: str = "base"):
    """
    Lazy-load Whisper model once per session.
    Sizes: tiny (39M) | base (74M) | small (244M) | medium (769M)
    'base' is the sweet spot: fast + accurate enough for most languages.
    """
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model


def transcribe_audio_bytes(audio_bytes: bytes, fmt: str = "wav") -> dict:
    """
    Transcribe audio using OpenAI Whisper (runs 100% locally).

    WHY WHISPER instead of Google Web Speech API:
    - Google Web Speech API is rate-limited and blocked on cloud servers
      (Streamlit Cloud, Heroku, etc.) — causes "Speech recognition failed:" errors
    - Whisper runs locally: no internet, no API key, no rate limits
    - Whisper supports 99 languages and auto-detects language
    - Whisper handles noisy audio better than Google STT

    Returns: {transcribed_text, detected_language}
    """
    try:
        import whisper
        import numpy as np
        from scipy.io import wavfile

        # Convert non-WAV to WAV via pydub if needed
        if fmt not in ("wav", "wave"):
            try:
                from pydub import AudioSegment
                seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
                buf = io.BytesIO()
                seg.export(buf, format="wav")
                audio_bytes = buf.getvalue()
            except Exception as exc:
                raise RuntimeError(f"Audio conversion failed: {exc}")

        # Write to temp file — Whisper needs a file path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            model = _get_whisper_model("base")
            # fp16=False for CPU-only environments (Streamlit Cloud uses CPU)
            result = model.transcribe(tmp_path, fp16=False)
            text = result["text"].strip()
            # Whisper returns ISO 639-1 language code directly
            whisper_lang = result.get("language", "unknown")
            # Map whisper lang codes to our standard codes
            lang_code = _whisper_to_iso(whisper_lang)
            return {"transcribed_text": text, "detected_language": lang_code}
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    except Exception as exc:
        raise RuntimeError(f"Transcription failed: {exc}") from exc


def _whisper_to_iso(whisper_lang: str) -> str:
    """Map Whisper language codes to ISO 639-1 codes used in our system."""
    mapping = {
        "chinese": "zh-cn",
        "zh": "zh-cn",
        "hebrew": "iw",
        "he": "iw",
    }
    return mapping.get(whisper_lang, whisper_lang)


# ════════════════════════════════════════════════════════════════
# EMBEDDINGS  —  sentence-transformers + FAISS
# ════════════════════════════════════════════════════════════════

_embed_model = None
_faiss_index = None
_faiss_texts: list[str] = []


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _embed(texts: list[str]):
    return _get_embed_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)


def chunk_text(text: str, size: int = 400, overlap: int = 60) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i: i + size])
        if chunk.strip():
            chunks.append(chunk.strip())
        i += size - overlap
    return chunks


def add_text_to_store(text: str) -> int:
    global _faiss_index, _faiss_texts
    try:
        import faiss
        chunks = chunk_text(text)
        if not chunks:
            return 0
        vecs = _embed(chunks).astype("float32")
        dim = vecs.shape[1]
        if _faiss_index is None:
            _faiss_index = faiss.IndexFlatIP(dim)
        _faiss_index.add(vecs)
        _faiss_texts.extend(chunks)
        return len(chunks)
    except Exception as exc:
        raise RuntimeError(f"Vector store failed: {exc}") from exc


def retrieve_context(query: str, top_k: int = 5) -> str:
    global _faiss_index, _faiss_texts
    if _faiss_index is None or _faiss_index.ntotal == 0:
        return ""
    try:
        import faiss
        q = _embed([query]).astype("float32")
        _, I = _faiss_index.search(q, min(top_k, _faiss_index.ntotal))
        return "\n\n---\n\n".join(_faiss_texts[i] for i in I[0] if i != -1)
    except Exception:
        return ""


def get_store_size() -> int:
    return _faiss_index.ntotal if _faiss_index else 0


def clear_vector_store():
    global _faiss_index, _faiss_texts
    _faiss_index = None
    _faiss_texts = []


# ════════════════════════════════════════════════════════════════
# GEMINI 2.5 FLASH  —  uses new google-genai SDK
# ════════════════════════════════════════════════════════════════

def ask_gemini(
    question: str,
    api_key: str,
    context: str = "",
    history: Optional[list[dict]] = None,
    response_language: str = "en",
) -> str:
    """
    Gemini 2.5 Flash with RAG context + multi-turn history.
    Uses the NEW google-genai SDK (pip install google-genai).
    Model: gemini-2.5-flash-preview-04-17  (stable preview, works on v1beta)
    """
    try:
        from google import genai
        from google.genai import types

        # Guard: never send empty content to Gemini
        if not question or not question.strip():
            raise ValueError("Question is empty — please record your voice and try again.")

        client = genai.Client(api_key=api_key)

        system_prompt = textwrap.dedent(f"""
            You are SignLingua Assistant — a multilingual AI embedded in an OCR
            and voice translation app.
            1. Answer the user's question clearly and helpfully.
            2. Use any provided context from sign boards or documents to ground answers.
            3. Respond in the language with ISO code: {response_language}
            4. Be concise — your answer will be spoken aloud via text-to-speech.
            5. If no context is available, answer from general knowledge.
        """).strip()

        # Build contents list: history + current question
        contents = []
        for turn in (history or []):
            role = "user" if turn["role"] == "user" else "model"
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=turn["content"])]
            ))

        # Compose current prompt with optional RAG context
        current_prompt = (
            f"[Retrieved context]\n{context}\n\n[Question]\n{question}"
            if context else question
        )
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=current_prompt)]
        ))

        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.4,
                top_p=0.9,
                max_output_tokens=800,
            ),
        )
        return response.text.strip()

    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Gemini API error: {exc}") from exc


# ════════════════════════════════════════════════════════════════
# FULL PIPELINES
# ════════════════════════════════════════════════════════════════

def image_pipeline(image_bytes: bytes, target_lang: str, add_to_store: bool = True) -> dict:
    """Image → OCR → Translate → TTS"""
    extracted = extract_text_from_image(image_bytes)
    tr = translate_text(extracted, target_lang)
    audio = text_to_speech(tr["translated_text"], target_lang)
    chunks_added = add_text_to_store(extracted) if add_to_store and extracted else 0
    src = tr["source_language"]
    return {
        "extracted_text": extracted,
        "source_language": src,
        "source_language_name": LANG_CODE_TO_NAME.get(src, src.upper()),
        "translated_text": tr["translated_text"],
        "audio_bytes": audio,
        "chunks_indexed": chunks_added,
    }


def voice_translate_pipeline(audio_bytes: bytes, target_lang: str) -> dict:
    """Live voice → Whisper STT → Detect → Translate → TTS"""
    stt = transcribe_audio_bytes(audio_bytes, fmt="wav")
    text = stt["transcribed_text"]
    if not text or not text.strip():
        raise ValueError("No speech detected. Please speak clearly and try again.")
    det = stt["detected_language"]
    tr = translate_text(text, target_lang, source=det)
    audio_out = text_to_speech(tr["translated_text"], target_lang)
    return {
        "transcribed_text": text,
        "detected_language": det,
        "detected_language_name": LANG_CODE_TO_NAME.get(det, det.upper()),
        "translated_text": tr["translated_text"],
        "audio_bytes": audio_out,
    }


def voice_qa_pipeline(
    audio_bytes: bytes,
    api_key: str,
    response_lang: str = "en",
    history: Optional[list[dict]] = None,
) -> dict:
    """Live voice → Whisper STT → RAG → Gemini 2.5 Flash → TTS"""
    stt = transcribe_audio_bytes(audio_bytes, fmt="wav")
    question = stt["transcribed_text"]
    if not question or not question.strip():
        raise ValueError("No speech detected. Please speak clearly and try again.")
    det = stt["detected_language"]
    context = retrieve_context(question, top_k=5)
    answer = ask_gemini(question, api_key, context, history, response_lang)
    audio_out = text_to_speech(answer, response_lang)
    return {
        "transcribed_question": question,
        "detected_language": det,
        "detected_language_name": LANG_CODE_TO_NAME.get(det, det.upper()),
        "context_used": context,
        "answer_text": answer,
        "answer_audio_bytes": audio_out,
    }
