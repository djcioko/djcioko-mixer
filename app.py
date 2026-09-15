import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import os
import tempfile
import unicodedata

st.set_page_config(page_title="SmartMix Pro V4.1", layout="wide")
st.title("🎧 SmartMix Pro - Manual Arrangement Studio")
st.caption(
    "BPM + key/Camelot, comenzi vocale Groq, taiere pe final de fraza, "
    "crossfade curat pe masuri (fără căderi de volum), export WAV/MP3."
)

# Salvarea stării pentru a nu pierde fișierele / setările la refresh
if "tracks" not in st.session_state:
    st.session_state.tracks = []
if "ultima_comanda" not in st.session_state:
    st.session_state.ultima_comanda = ""
if "ultimul_rezultat" not in st.session_state:
    st.session_state.ultimul_rezultat = ""
if "mic_id" not in st.session_state:
    st.session_state.mic_id = 0

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
    best_score, best_idx, best_mode = -1e9, 0, "major"
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
    y, sr = librosa.load(path, sr=22050, duration=90)
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
            return camelot_distance(last_c, t.get("Camelot", "?")) + abs(
                float(t.get("BPM", 120) or 120) - last_bpm
            ) / 8.0

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
        raise RuntimeError("Lipseste GROQ_API_KEY (Secrets sau variabila de mediu).")
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
        return "Nu sunt piese in lista. Analizeaza intai fisierele."
    if any(w in t for w in ("goleste", "sterge lista", "reseteaza")):
        st.session_state.tracks = []
        return "Lista a fost golita."
    if any(w in t for w in ("armonic", "compatibil", "armonie")):
        st.session_state.tracks = harmonic_order(st.session_state.tracks)
        return "Am aranjat armonic (key + BPM)."
    if any(
        w in t
        for w in (
            "camelot", "kamelot", "kameleot", "cameleot", "cameleon", "kameleon",
            "tonalitate", "tonalitat", "tonalita", "key", "nota",
        )
    ):
        def cam_key(track):
            p = parse_camelot(track.get("Camelot", "?"))
            return p if p else (99, "Z")

        st.session_state.tracks = sorted(
            st.session_state.tracks, key=lambda x: (cam_key(x), x["BPM"])
        )
        return "Am sortat dupa Camelot / tonalitate."
    if "bpm" in t or "tempo" in t:
        desc = any(w in t for w in ("mare", "descresc", "invers", "cobor", "de sus"))
        st.session_state.tracks = sorted(
            st.session_state.tracks, key=lambda x: x["BPM"], reverse=desc
        )
        return "Am sortat BPM de la mare la mic." if desc else "Am sortat BPM de la mic la mare."
    return "Nu am recunoscut comanda. Incearca: sorteaza dupa BPM, aranjeaza pe Camelot, goleste lista."


def vocal_band_energy(y, sr, hop=512):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mask = (freqs >= 200) & (freqs <= 4000)
    band = S[mask].mean(axis=0)
    return band


def find_phrase_end(y, sr, target_sec, window_sec=12.0):
    dur = len(y) / float(sr)
    target_sec = min(max(target_sec, 4.0), max(dur - 1.0, 1.0))
    t0 = max(0.0, target_sec - window_sec)
    t1 = min(dur, target_sec + window_sec)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    bpm = float(np.asarray(tempo).item()) if np.size(tempo) else 120.0
    if bpm <= 1:
        bpm = 120.0

    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    voc = vocal_band_energy(y, sr, hop=hop)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    cands = beats[(beats >= t0) & (beats <= t1)] if beats is not None and len(beats) else np.array([])
    if cands.size == 0:
        step = 60.0 / bpm
        cands = np.arange(t0, t1, step)

    def frame_at(t):
        return int(np.clip(np.searchsorted(times, t), 0, len(rms) - 1))

    rms_n = rms / (np.percentile(rms, 90) + 1e-8)
    voc_n = voc / (np.percentile(voc, 90) + 1e-8)

    best_t, best_score = target_sec, 1e9
    for t in cands:
        i = frame_at(t)
        j = min(len(rms) - 1, i + max(1, int(0.75 * sr / hop)))
        energy = float(np.mean(rms_n[i:j]))
        voice = float(np.mean(voc_n[i:j]))
        late = 0.015 * max(0.0, target_sec - t)
        early = 0.04 * max(0.0, t - target_sec)
        score = 1.4 * voice + 1.0 * energy + late + early
        if score < best_score:
            best_score, best_t = score, float(t)

    return best_t, bpm


def bars_to_samples(bpm, bars, sr, beats_per_bar=4):
    beats = max(1, int(bars * beats_per_bar))
    sec = beats * (60.0 / max(bpm, 40.0))
    sec = float(np.clip(sec, 1.5, 12.0))
    return int(sec * sr), sec


def smooth_crossfade(out_seg, in_seg):
    """Realizează un crossfade uniform, curat, prevenind căderea bruscă de volum."""
    n = min(len(out_seg), len(in_seg))
    out_seg, in_seg = out_seg[:n], in_seg[:n]
    
    # Curbă logaritmică / sinuzoidală sau liniară fină pentru un power-curve crossfade egal (egalizator de putere)
    t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
    fade_out = np.cos(t)
    fade_in = np.sin(t)
    
    mixed = out_seg * fade_out + in_seg * fade_in
    return mixed


st.sidebar.header("Setari Mix")
durata_def = st.sidebar.number_input("Durata tinta piese (sec):", 30, 600, 120)
bars_xf = st.sidebar.selectbox("Crossfade (masuri):", [1, 2, 4], index=2) # 4 măsuri default
win_frase = st.sidebar.slider("Cauta final de strofa +/- sec:", 4, 20, 12)
fmt_out = st.sidebar.selectbox("Format:", ["WAV", "MP3 320kbps"])
st.sidebar.caption("Crossfade-ul pe masuri asigura trecerea lina intre piese.")

st.sidebar.header("Voce")
if st.sidebar.button("Reinregistreaza", key="reset_mic_side"):
    st.session_state.mic_id += 1
    st.rerun()

st.markdown("### Comanda vocala")
v1, v2, v3 = st.columns([2, 1, 1])
with v1:
    audio_cmd = st.audio_input("Inregistreaza comanda", key=f"mic_{st.session_state.mic_id}")
with v2:
    run_voice = st.button("Executa comanda vocala", type="primary")
with v3:
    if st.button("Reinregistreaza"):
        st.session_state.mic_id += 1
        st.rerun()

if audio_cmd is not None and run_voice:
    try:
        text = transcrie_groq(
            audio_cmd.getvalue(), filename=getattr(audio_cmd, "name", None) or "comanda.wav"
        )
        st.session_state.ultima_comanda = text
        st.session_state.ultimul_rezultat = executa_comanda(text)
        st.session_state.mic_id += 1
        st.rerun()
    except Exception as e:
        st.error(str(e))

if st.session_state.ultima_comanda:
    st.success(f"Auzit: {st.session_state.ultima_comanda}")
    st.info(st.session_state.ultimul_rezultat)

files = st.file_uploader(
    "Incarca muzica:", type=["mp3", "wav", "ogg", "flac", "m4a"], accept_multiple_files=True
)
if files and st.button("ANALIZEAZA PIESELE", type="primary"):
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

if st.session_state.tracks:
    st.markdown("### Lista de Mixaj")
    c1, c2, c3, c4, c5 = st.columns([1.4, 1.4, 1.6, 1.8, 1.2])
    if c1.button("BPM Mic->Mare"):
        st.session_state.tracks = sorted(st.session_state.tracks, key=lambda x: x["BPM"])
        st.rerun()
    if c2.button("BPM Mare->Mic"):
        st.session_state.tracks = sorted(st.session_state.tracks, key=lambda x: x["BPM"], reverse=True)
        st.rerun()
    if c3.button("Sorteaza Camelot"):
        def cam_key(t):
            p = parse_camelot(t.get("Camelot", "?"))
            return p if p else (99, "Z")

        st.session_state.tracks = sorted(st.session_state.tracks, key=lambda t: (cam_key(t), t["BPM"]))
        st.rerun()
    if c4.button("Aranjare armonica"):
        st.session_state.tracks = harmonic_order(st.session_state.tracks)
        st.rerun()
    if c5.button("Goleste"):
        st.session_state.tracks = []
        st.rerun()

    for i, track in enumerate(st.session_state.tracks):
        col_move, col_info, col_edit = st.columns([1, 4, 3])
        with col_move:
            if st.button("UP", key=f"up_{i}") and i > 0:
                st.session_state.tracks[i], st.session_state.tracks[i - 1] = (
                    st.session_state.tracks[i - 1],
                    st.session_state.tracks[i],
                )
                st.rerun()
            if st.button("DOWN", key=f"down_{i}") and i < len(st.session_state.tracks) - 1:
                st.session_state.tracks[i], st.session_state.tracks[i + 1] = (
                    st.session_state.tracks[i + 1],
                    st.session_state.tracks[i],
                )
                st.rerun()
        with col_info:
            st.markdown(f"**{i + 1}. {track['Piesa']}**")
            st.caption(
                f"BPM: {track['BPM']}  |  Key: {track.get('Key', '?')} "
                f"({track.get('Camelot', '?')})  |  Start: {track['Start (sec)']}s"
            )
            if track.get("Eroare"):
                st.warning(track["Eroare"])
        with col_edit:
            st.session_state.tracks[i]["Start (sec)"] = st.number_input(
                "Start (s)", value=float(track["Start (sec)"]), key=f"s_{i}", step=0.1
            )
            st.session_state.tracks[i]["Durata (sec)"] = st.number_input(
                "Durata tinta (s)", value=float(track["Durata (sec)"]), key=f"d_{i}", step=1.0
            )

    st.markdown("---")
    if st.button("GENEREAZA MIXUL FINAL", type="primary"):
        try:
            with st.spinner("Caut finaluri de fraza si mixez curat pe masuri..."):
                sr_mix = 44100
                final = None
                log_rows = []

                for i, row in enumerate(st.session_state.tracks):
                    src = row.get("Path") or row["Piesa"]
                    start = float(row["Start (sec)"])
                    tinta = float(row["Durata (sec)"])
                    load_dur = tinta + win_frase + 8
                    y, _ = librosa.load(src, sr=sr_mix, offset=max(0.0, start), duration=load_dur)
                    if y.size < sr_mix:
                        raise ValueError(f"Piesa prea scurta: {row['Piesa']}")
                    y = librosa.util.normalize(y).astype(np.float32) * 0.95

                    out_t, bpm_loc = find_phrase_end(y, sr_mix, tinta, window_sec=float(win_frase))
                    bpm_use = float(row.get("BPM") or bpm_loc or 120)
                    fade_n, fade_sec = bars_to_samples(bpm_use, int(bars_xf), sr_mix)

                    cut = int(out_t * sr_mix)
                    cut = max(fade_n + 1, min(cut, len(y)))
                    piece = y[:cut]

                    log_rows.append(
                        f"{i + 1}. {row['Piesa']}: tinta {tinta:.0f}s -> taiat la {out_t:.2f}s "
                        f"(crossfade {fade_sec:.2f}s / {bars_xf} masuri)"
                    )

                    if final is None:
                        final = piece
                    else:
                        fade_n = min(fade_n, len(final), len(piece))
                        if fade_n < 64:
                            final = np.concatenate([final, piece])
                        else:
                            # Aplicăm noul crossfade lin fără scăderi bruște
                            mixed = smooth_crossfade(final[-fade_n:], piece[:fade_n])
                            final = np.concatenate([final[:-fade_n], mixed, piece[fade_n:]])

                out_name = f"SmartMix_Result.{'mp3' if 'MP3' in fmt_out else 'wav'}"
                if "MP3" in fmt_out:
                    sf.write("t.wav", final, sr_mix)
                    rc = os.system(f'ffmpeg -i "t.wav" -ab 320k -y "{out_name}"')
                    if rc != 0 or not os.path.exists(out_name):
                        raise RuntimeError("ffmpeg a esuat. Exporta WAV sau instaleaza ffmpeg.")
                else:
                    sf.write(out_name, final, sr_mix, subtype="PCM_24")

                st.success("Mix gata.")
                for line in log_rows:
                    st.caption(line)
                st.audio(out_name)
                with open(out_name, "rb") as f_res:
                    st.download_button("DESCARCA MIXUL", f_res, file_name=out_name)
        except Exception as e:
            st.error(f"Eroare: {e}")
else:
    st.write("Incarca piese, analizeaza, apoi genereaza mixul.")
