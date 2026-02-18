import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import tempfile

st.set_page_config(page_title="SmartMix Pro V4.0 - Pro Transition", layout="wide")
st.title("🥁 SmartMix Pro: Redrum Glue Edition")

if 'tracks' not in st.session_state: st.session_state.tracks = []
if 'drum_slots' not in st.session_state: st.session_state.drum_slots = {}

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

# --- UPLOAD MELODII ---
files = st.file_uploader("Încarcă melodiile:", type=['mp3', 'wav'], accept_multiple_files=True)
if files:
    for f in files:
        if not any(t['nume'] == f.name for t in st.session_state.tracks):
            t = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            t.write(f.getbuffer())
            st.session_state.tracks.append({
                "nume": f.name, "path": t.name, "drum_loop": "Fără", "durata": 60
            })
    st.rerun()

# --- CONFIGURARE ---
if st.session_state.tracks:
    for i, track in enumerate(st.session_state.tracks):
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1: st.write(f"**{i+1}. {track['nume']}**")
            with c2:
                opts = ["Fără"] + list(st.session_state.drum_slots.keys())
                st.session_state.tracks[i]['drum_loop'] = st.selectbox(f"Tobe spre următoarea:", opts, key=f"d_{i}")
            with c3: st.session_state.tracks[i]['durata'] = st.number_input("Secunde:", 5, 600, 60, key=f"s_{i}")

    if st.button("🚀 GENEREAZĂ MIXUL CU TRANZIȚIE LIPSITĂ", type="primary"):
        with st.spinner("Se lipesc piesele cu Redrum..."):
            sr = 44100
            final_audio = None
            
            for i, row in enumerate(st.session_state.tracks):
                # Încărcăm piesa curentă
                y, _ = librosa.load(row['path'], sr=sr, mono=True, duration=row['durata'])
                y = librosa.util.normalize(y)

                # Dacă avem tobe, le aplicăm la finalul piesei curente
                if row['drum_loop'] != "Fără":
                    y_d, _ = librosa.load(st.session_state.drum_slots[row['drum_loop']], sr=sr, mono=True)
                    ov_len = min(len(y_d), int(8 * sr), len(y))
                    # Mixăm tobele peste finalul piesei 1
                    y[-ov_len:] = (y[-ov_len:] * 0.4) + (y_d[:ov_len] * 0.6)
                
                if final_audio is None:
                    final_audio = y
                else:
                    # AICI E SECRETUL: Suprapunem piesa 2 peste finalul piesei 1
                    # Folosim o zonă de overlap de 4 secunde pentru "lipire"
                    overlap_sec = 4 
                    ov_samples = int(overlap_sec * sr)
                    
                    # Facem un mic fade-in pentru piesa care intră
                    fade_in = np.linspace(0, 1, ov_samples)
                    y_start = y[:ov_samples] * fade_in
                    
                    # Combinăm finalul mixului de până acum cu începutul piesei noi
                    mixed_zone = final_audio[-ov_samples:] + y_start
                    final_audio = np.concatenate([final_audio[:-ov_samples], mixed_zone, y[ov_samples:]])

            sf.write("mix_pro.wav", final_audio, sr)
            st.audio("mix_pro.wav")
            st.download_button("💾 DESCARCĂ MIXUL", open("mix_pro.wav", "rb"), "Mix_Pro_Redrum.wav")
