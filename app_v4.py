import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import tempfile

st.set_page_config(page_title="SmartMix V5 - Drum Library", layout="wide")
st.title("🎧 SmartMix V5: Automatic Drum Matcher")

if 'tracks' not in st.session_state: st.session_state.tracks = []
if 'drum_lib' not in st.session_state: st.session_state.drum_lib = []

# --- 1. UPLOAD LIBRARIE TOBE (Cele 50 de loop-uri) ---
with st.sidebar:
    st.header("🥁 Librăria de Tobele Tale")
    up_drums = st.file_uploader("Urci aici toate loop-urile (12-15s):", type=['mp3', 'wav'], accept_multiple_files=True)
    
    if up_drums and not st.session_state.drum_lib:
        with st.spinner("Analizez librăria de tobe..."):
            for d in up_drums:
                t = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                t.write(d.getbuffer())
                # Analizăm rapid BPM-ul fiecărui loop urcat
                y_d, sr = librosa.load(t.name, duration=10)
                tempo, _ = librosa.beat.beat_track(y=y_d, sr=sr)
                bpm = float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo)
                
                st.session_state.drum_lib.append({
                    "nume": d.name, "path": t.name, "bpm": round(bpm, 1)
                })
        st.success(f"Librărie gata: {len(st.session_state.drum_lib)} loop-uri.")

# --- 2. UPLOAD MELODII ---
files = st.file_uploader("Încarcă melodiile pentru mix:", type=['mp3', 'wav'], accept_multiple_files=True)
if files:
    for f in files:
        if not any(t['nume'] == f.name for t in st.session_state.tracks):
            t = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            t.write(f.getbuffer())
            
            # Analizăm BPM-ul melodiei ca să știm ce tobe să căutăm
            y_p, sr = librosa.load(t.name, duration=20)
            tempo_p, _ = librosa.beat.beat_track(y=y_p, sr=sr)
            bpm_p = float(tempo_p[0]) if isinstance(tempo_p, (np.ndarray, list)) else float(tempo_p)
            
            # CĂUTARE AUTOMATĂ: Găsim toba cu cel mai apropiat BPM
            best_drum = "Fără"
            if st.session_state.drum_lib:
                # Găsește loop-ul care are diferența de BPM cea mai mică
                closest = min(st.session_state.drum_lib, key=lambda x: abs(x['bpm'] - bpm_p))
                best_drum = closest['nume']

            st.session_state.tracks.append({
                "nume": f.name, "path": t.name, "bpm": round(bpm_p, 1),
                "drum_loop": best_drum, "durata": 60
            })
    st.rerun()

# --- 3. CONFIGURARE ȘI GENERARE ---
if st.session_state.tracks:
    st.write("### Verifică Sugestiile Automate")
    for i, track in enumerate(st.session_state.tracks):
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 1])
            with c1: st.write(f"**{track['nume']}** (BPM: {track['bpm']})")
            with c2:
                opts = ["Fără"] + [d['nume'] for d in st.session_state.drum_lib]
                st.session_state.tracks[i]['drum_loop'] = st.selectbox(f"Tobe alese:", opts, index=opts.index(track['drum_loop']), key=f"s_{i}")
            with c3: st.session_state.tracks[i]['durata'] = st.number_input("Sec:", 10, 600, 60, key=f"t_{i}")

    if st.button("🚀 GENEREAZĂ MIXUL SMART", type="primary"):
        sr = 44100
        final_audio = None
        for i, row in enumerate(st.session_state.tracks):
            y, _ = librosa.load(row['path'], sr=sr, mono=True, duration=row['durata'])
            
            # Aplicăm loop-ul de tobe ales automat
            if row['drum_loop'] != "Fără":
                d_info = next(item for item in st.session_state.drum_lib if item["nume"] == row['drum_loop'])
                y_d, _ = librosa.load(d_info['path'], sr=sr, mono=True)
                ov = min(len(y_d), int(10 * sr), len(y))
                y[-ov:] = (y[-ov:] * 0.4) + (y_d[:ov] * 0.6) # Layering Redrum

            if final_audio is None: final_audio = y
            else:
                # Lipire cu overlap de 6 secunde pentru fluiditate
                ov_s = int(6 * sr)
                mixed = final_audio[-ov_s:] + (y[:ov_s] * np.linspace(0, 1, ov_s))
                final_audio = np.concatenate([final_audio[:-ov_s], mixed, y[ov_s:]])

        sf.write("smart_mix.wav", final_audio, sr)
        st.audio("smart_mix.wav")
