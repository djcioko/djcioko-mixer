import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import os
import tempfile
import unicodedata

st.set_page_config(page_title="SmartMix Pro V3.9", layout="wide")
st.title("🎧 SmartMix Pro - Manual Arrangement Studio")
st.caption("Analiză BPM + tonalitate (key / Camelot), sortare armonică, comenzi vocale (Groq Whisper), mix WAV/MP3.")

if "tracks" not in st.session_state:
    st.session_state.tracks = []
if "ultima_comanda" not in st.session_state:
    st.session_state.ultima_comanda = ""
if "ultimul_rezultat" not in st.session_state:
    st.session_state.ultimul_rezultat = ""

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CAMELOT_MAJOR = {
    0: "8B", 1: "3B", 2: "10B", 3: "5B", 4: "12B", 5: "7B",
    6: "2B", 7: "9B", 8: "4B", 9: "11B", 10: "6B", 11: "1B",
}
CAMELOT_MINOR = {
    0: "5A", 1: "12A", 2: "7A", 3: "2A", 4: "9A", 5: "4A",
    6: "11A", 7: "6A", 8: "1A", 9: "8A", 10: "3A", 11: "10A",
}

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def parse_camelot(code):
    if not code or code == "?":
        return None
    try:
        n = int(code[:-1])
        letter = code[-1].upper()
        if 1 <= n <= 12 and letter in ("A", "B"):
            return n, letter
    except Exception:
        return None
    return None


def camelot_distance(a, b):
    pa, pb = parse_camelot(a), parse_camelot(b)
    if not pa or not pb:
        return 99
    na, la = pa
    nb, lb = pb
    ring = min((na - nb) % 12, (nb - na) % 12)
    if la == lb:
        return ring
    if ring == 0:
        return 1
    return ring + 1


def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.maximum(chroma.mean(axis=1), 1e-8)
    chroma_mean = chroma_mean / (chroma_mean.sum() + 1e-8)

    best_score = -1e9
    best_idx = 0
    best_mode = "major"

    for i in range(12):
        rolled = np.roll(chroma_mean, -i)
        s_maj = float(np.corrcoef(rolled, MAJOR_PROFILE)[0, 1])
        s_min = float(np.corrcoef(rolled, MINOR_PROFILE)[0, 1])
        if np.isnan(s_maj):
            s_maj = -1e9
        if np.isnan(s_min):
            s_min = -1e9
        if s_maj > best_score:
            best_score, best_idx, best_mode = s_maj, i, "major"
        if s_min > best_score:
            best_score, best_idx, best_mode = s_min, i, "minor"

    if best_mode == "major":
        return NOTES[best_idx], CAMELOT_MAJOR[best_idx]
    return NOTES[best_idx] + "m", CAMELOT_MINOR[best_idx]


def analyze_track(path):
    y, sr = librosa.load(path, sr=22050, duration=60)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)[0]
    thr = float(np.mean(rmse) * 1.3)
    hits = np.where(rmse > thr)[0]
    start_s = float(librosa.frames_to_time(hits[0], sr=sr)) if len(hits) else 0.0
    key, camelot = detect_key(y, sr)
    return round(float(np.asarray(tempo).item()), 1), round(start_s, 2), key, camelot


def harmonic_order(tracks):
    if not tracks:
        return tracks
    remaining = tracks[:]
    ordered = [remaining.pop(0)]
    while remaining:
        last = ordered[-1]
        last_c = last.get("Camelot", "?")
        last_bpm = float(last.get("BPM", 120) or 120)

        def score(t):
            d_key = camelot_distance(last_c, t.get("Camelot", "?"))
            d_bpm = abs(float(t.get("BPM", 120) or 120) - last_bpm) / 8.0
            return d_key + d_bpm

        remaining.sort(key=score)
        ordered.append(remaining.pop(0))
    return ordered


def save_upload(uploaded):
    suffix = os.path.splitext(uploaded.name)[1] or ".mp3"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name


def fara_diacritice(text):
    text = unicodedata.normalize("NFD", text or "")
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn").lower()


def transcrie_groq(audio_bytes, filename="comanda.wav"):
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "Lipsește GROQ_API_KEY. Pune-o în variabila de mediu sau în Streamlit Secrets."
        )
    client = Groq(api_key=api_key)
    transcription = client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model="whisper-large-v3-turbo",
        language="ro",
        temperature=0,
    )
    return (transcription.text or "").strip()


def executa_comanda(text):
    t = fara_diacritice(text)
    if not st.session_state.tracks:
        return "Nu sunt piese în listă. Analizează întâi fișierele."

    if any(w in t for w in ("goleste", "sterge lista", "reseteaza")):
        st.session_state.tracks = []
        return "Lista a fost golită."

    if any(w in t for w in ("armonic", "compatibil", "armonie")):
        st.session_state.tracks = harmonic_order(st.session_state.tracks)
        return "Am aranjat armonic (key + BPM)."

    if any(w in t for w in ("camelot", "kamelot", "tonalitate", "tonalitat", "key", "nota")):
        def cam_key(track):
            p = parse_camelot(track.get("Camelot", "?"))
            return p if p else (99, "Z")
        st.session_state.tracks = sorted(
            st.session_state.tracks, key=lambda x: (cam_key(x), x["BPM"])
        )
        return "Am sortat după Camelot / tonalitate."

    if "bpm" in t or "tempo" in t:
        desc = any(w in t for w in ("mare", "descresc", "invers", "cobor", "de sus"))
        st.session_state.tracks = sorted(
            st.session_state.tracks, key=lambda x: x["BPM"], reverse=desc
        )
        return "Am sortat BPM de la mare la mic." if desc else "Am sortat BPM de la mic la mare."

    return (
        "Nu am recunoscut comanda. Încearcă: "
        "„sortează după BPM”, „aranjează pe Camelot”, „aranjare armonică”, „golește lista”."
    )


# --- SIDEBAR ---
st.sidebar.header("🚀 Setări Mix")
durata_def = st.sidebar.number_input("Durată standard piese (sec):", 30, 600, 120)
cf_sec = st.sidebar.slider("Crossfade (sec):", 2, 15, 5)
fmt_out = st.sidebar.selectbox("Format:", ["WAV", "MP3 320kbps"])

st.sidebar.header("🎙️ Comandă vocală")
st.sidebar.write("Ex: „sortează după BPM”, „aranjează pe Camelot”, „aranjare armonică”.")
audio_cmd = st.sidebar.audio_input("Înregistrează comanda")
if audio_cmd is not None and st.sidebar.button("▶ Execută comanda vocală", type="primary"):
    try:
        text = transcrie_groq(audio_cmd.getvalue(), filename=audio_cmd.name or "comanda.wav")
        st.session_state.ultima_comanda = text
        st.session_state.ultimul_rezultat = executa_comanda(text)
        st.rerun()
    except Exception as e:
        st.sidebar.error(str(e))

if st.session_state.ultima_comanda:
    st.sidebar.success(f"Auzit: {st.session_state.ultima_comanda}")
    st.sidebar.info(st.session_state.ultimul_rezultat)

# --- UPLOAD ---
files = st.file_uploader(
    "Încarcă muzica:",
    type=["mp3", "wav", "ogg", "flac", "m4a"],
    accept_multiple_files=True,
)

if files:
    if st.button("🔍 ANALIZEAZĂ PIESELE", type="primary"):
        progress = st.progress(0)
        for i, f in enumerate(files):
            if any(t["Piesa"] == f.name for t in st.session_state.tracks):
                progress.progress((i + 1) / len(files))
                continue
            path = save_upload(f)
            try:
                bpm, s_start, key, camelot = analyze_track(path)
                st.session_state.tracks.append(
                    {
                        "Piesa": f.name,
                        "Path": path,
                        "BPM": bpm,
                        "Key": key,
                        "Camelot": camelot,
                        "Start (sec)": s_start,
                        "Durata (sec)": float(durata_def),
                    }
                )
            except Exception as e:
                st.session_state.tracks.append(
                    {
                        "Piesa": f.name,
                        "Path": path,
                        "BPM": 120.0,
                        "Key": "?",
                        "Camelot": "?",
                        "Start (sec)": 0.0,
                        "Durata (sec)": float(durata_def),
                        "Eroare": str(e),
                    }
                )
            progress.progress((i + 1) / len(files))
        st.rerun()

# --- LISTA ---
if st.session_state.tracks:
    st.markdown("### 📋 Lista de Mixaj (Aranjează Ordinea)")

    c1, c2, c3, c4, c5 = st.columns([1.4, 1.4, 1.6, 1.8, 1.2])
    if c1.button("📉 BPM Mic→Mare"):
        st.session_state.tracks = sorted(st.session_state.tracks, key=lambda x: x["BPM"])
        st.rerun()
    if c2.button("📈 BPM Mare→Mic"):
        st.session_state.tracks = sorted(st.session_state.tracks, key=lambda x: x["BPM"], reverse=True)
        st.rerun()
    if c3.button("🎹 Sortează Camelot"):
        def cam_key(t):
            p = parse_camelot(t.get("Camelot", "?"))
            return p if p else (99, "Z")
        st.session_state.tracks = sorted(st.session_state.tracks, key=lambda t: (cam_key(t), t["BPM"]))
        st.rerun()
    if c4.button("🧬 Aranjare armonică"):
        st.session_state.tracks = harmonic_order(st.session_state.tracks)
        st.rerun()
    if c5.button("🗑️ Golește"):
        st.session_state.tracks = []
        st.rerun()

    for i, track in enumerate(st.session_state.tracks):
        col_move, col_info, col_edit = st.columns([1, 4, 3])
        with col_move:
            if st.button("🔼", key=f"up_{i}") and i > 0:
                st.session_state.tracks[i], st.session_state.tracks[i - 1] = (
                    st.session_state.tracks[i - 1],
                    st.session_state.tracks[i],
                )
                st.rerun()
            if st.button("🔽", key=f"down_{i}") and i < len(st.session_state.tracks) - 1:
                st.session_state.tracks[i], st.session_state.tracks[i + 1] = (
                    st.session_state.tracks[i + 1],
                    st.session_state.tracks[i],
                )
                st.rerun()
        with col_info:
            st.markdown(f"**{i + 1}. {track['Piesa']}**")
            st.caption(
                f"BPM: {track['BPM']}  |  Key: {track.get('Key', '?')}  "
                f"({track.get('Camelot', '?')})  |  Start: {track['Start (sec)']}s"
            )
            if track.get("Eroare"):
                st.warning(f"Analiză incompletă: {track['Eroare']}")
        with col_edit:
            new_start = st.number_input("Start (s)", value=float(track["Start (sec)"]), key=f"s_{i}", step=0.1)
            new_dur = st.number_input("Durată (s)", value=float(track["Durata (sec)"]), key=f"d_{i}", step=1.0)
            st.session_state.tracks[i]["Start (sec)"] = new_start
            st.session_state.tracks[i]["Durata (sec)"] = new_dur

    st.markdown("---")
    if st.button("🚀 GENEREAZĂ MIXUL FINAL", type="primary"):
        try:
            with st.spinner("Mixare profesională în curs..."):
                sr_mix = 44100
                final = np.array([], dtype=np.float32)
                for i, row in enumerate(st.session_state.tracks):
                    src = row.get("Path") or row["Piesa"]
                    y, _ = librosa.load(
                        src,
                        sr=sr_mix,
                        offset=float(row["Start (sec)"]),
                        duration=float(row["Durata (sec)"]),
                    )
                    if y.size == 0:
                        raise ValueError(f"Piesa goală: {row['Piesa']}")
                    y = librosa.util.normalize(y).astype(np.float32) * 0.95
                    f_len = int(cf_sec * sr_mix)
                    if i == 0:
                        final = y
                    else:
                        f_len = min(f_len, len(final), len(y))
                        if f_len <= 0:
                            final = np.concatenate([final, y])
                        else:
                            out_f = final[-f_len:] * np.linspace(1, 0, f_len, dtype=np.float32)
                            in_f = y[:f_len] * np.linspace(0, 1, f_len, dtype=np.float32)
                            final[-f_len:] = out_f + in_f
                            final = np.concatenate([final, y[f_len:]])

                out_name = f"SmartMix_Result.{'mp3' if 'MP3' in fmt_out else 'wav'}"
                if "MP3" in fmt_out:
                    sf.write("t.wav", final, sr_mix)
                    rc = os.system(f'ffmpeg -i "t.wav" -ab 320k -y "{out_name}"')
                    if rc != 0 or not os.path.exists(out_name):
                        raise RuntimeError("ffmpeg a eșuat. Exportă WAV sau instalează ffmpeg.")
                else:
                    sf.write(out_name, final, sr_mix, subtype="PCM_24")
                st.success("Mix gata.")
                st.audio(out_name)
                with open(out_name, "rb") as f_res:
                    st.download_button("⬇️ DESCARCĂ MIXUL", f_res, file_name=out_name)
        except Exception as e:
            st.error(f"Eroare: {e}")
else:
    st.write("Încarcă fișiere audio și apasă **ANALIZEAZĂ PIESELE**. Comanda vocală e în sidebar.")
