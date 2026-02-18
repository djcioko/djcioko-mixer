import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import os
import tempfile

st.set_page_config(page_title="SmartMix Pro V4.0 - Fast Edition", layout="wide")

# --- INITIALIZARE ---
if 'tracks' not in st.session_state:
    st.session_state.tracks = []
if 'drum_slots' not in st.session_state:
    st.session_state.drum_slots = {}

st.title("🎧 SmartMix Pro V4.0")

# --- SIDEBAR: DRUM SLOTS ---
with st.sidebar:
    st.header("🥁 Drum Slots")
    up_drums = st.file_uploader("Încarcă tobe (8-12s):", type=['mp3', 'wav'], accept_multiple_files=True)
    if up_drums:
        for d in up_drums:
            if d.name not in st.session_state.drum_slots:
                t = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                t.write(d.getbuffer())
                st.session_state.drum_slots[d.name] = t.name
        st.success(f"Tobe active: {len(st.session_state.drum_slots)}")

# --- UPLOAD MELODII ---
files = st.file_uploader("Încarcă melodiile:", type=['mp3', 'wav'], accept_multiple_files=True)

if files:
    new_added = False
    for f in files:
        if not any(t['nume'] == f.name for t in st.session_state.tracks):
            t = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            t.write(f.getbuffer())
            # Adăugăm piesa FĂRĂ analiză BPM ca să fie instantaneu
            st.session_state.tracks.append({
                "nume": f.name, "path": t.name, "bpm": 120.0,
                "drum_loop": "Fără (Crossfade)", "durata": 60, "start": 0
            })
            new_added = True
    if new_added:
        st.rerun()

# --- INTERFATA CONFIGURARE ---
if len(st.session_state.tracks) > 0:
    st.markdown("### 📋 Configurarea Mixajului")
    
    for i, track in enumerate(st.session_state.tracks):
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                st.write(f"**{i+1}. {track['nume']}**")
            with c2:
                opts = ["Fără (Crossfade)"] + list(st.session_state.drum_slots.keys())
                st.session_state.tracks[i]['drum_loop'] = st.selectbox(f"Tobe final:", opts, key=f"d_{i}")
            with c3:
                st.session_state.tracks[i]['durata'] = st.number_input("Secunde:", 5, 600, 60, key=f"s_{i}")

    if st.button("🚀 GENEREAZĂ MIXUL FINAL", type="primary"):
        with st.spinner("Se amestecă piesele..."):
            sr = 44100
            final_audio = np.array([], dtype=np.float32)
            for i, row in enumerate(st.session_state.tracks):
                y, _ = librosa.load(row['path'], sr=sr, mono=True, duration=row['durata'])
                y = librosa.util.normalize(y) * 0.8
                
                # Redrum Layer
                if row['drum_loop'] != "Fără (Crossfade)":
                    y_d, _ = librosa.load(st.session_state.drum_slots[row['drum_loop']], sr=sr, mono=True)
                    ov = min(len(y_d), int(8 * sr), len(y))
                    y[-ov:] = (y[-ov:] * 0.5) + (y_d[:ov] * 0.6)
                
                if i == 0: final_audio = y
                else: final_audio = np.concatenate([final_audio, y])
            
            sf.write("mix.wav", final_audio, sr)
            st.audio("mix.wav")
            st.download_button("💾 DESCARCĂ", open("mix.wav", "rb"), "Mix_Final.wav")
else:
    st.info("Încarcă piese pentru a vedea setările.")
