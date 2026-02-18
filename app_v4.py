import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import tempfile

st.set_page_config(page_title="SmartMix V5 - Fast Library", layout="wide")
st.title("🎧 SmartMix Pro: Instant Library Mode")

# Initializare liste
if 'tracks' not in st.session_state: st.session_state.tracks = []
if 'drum_lib' not in st.session_state: st.session_state.drum_lib = {}

# --- SIDEBAR: LIBRARIA TA DE TOBE ---
with st.sidebar:
    st.header("🥁 Librăria de Tobele Tale")
    st.write("Încarcă aici cele 50 de loop-uri.")
    up_drums = st.file_uploader("Încarcă tobe:", type=['mp3', 'wav'], accept_multiple_files=True)
    
    if up_drums:
        for d in up_drums:
            if d.name not in st.session_state.drum_lib:
                t = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                t.write(d.getbuffer())
                st.session_state.drum_lib[d.name] = t.name
        st.success(f"Librărie activă: {len(st.session_state.drum_lib)} loop-uri")

# --- ZONA DE UPLOAD MELODII ---
files = st.file_uploader("Încarcă melodiile principale:", type=['mp3', 'wav'], accept_multiple_files=True)

if files:
    new_data = False
    for f in files:
        if not any(t['nume'] == f.name for t in st.session_state.tracks):
            t = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            t.write(f.getbuffer())
            # Adăugăm piesa INSTANT, fără analiză BPM
            st.session_state.tracks.append({
                "nume": f.name, "path": t.name, "drum_loop": "Fără", "durata": 60
            })
            new_data = True
    if new_data:
        st.rerun()

# --- CONFIGURARE INTERFAȚĂ (Apare imediat!) ---
if st.session_state.tracks:
    st.write("### 📋 Configurează Mixul")
    for i, track in enumerate(st.session_state.tracks):
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 1])
            with c1: st.write(f"**{i+1}. {track['nume']}**")
            with c2:
                # Selector manual din toată librăria ta
                opts = ["Fără"] + list(st.session_state.drum_lib.keys())
                st.session_state.tracks[i]['drum_loop'] = st.selectbox(f"Alege toba de tranziție:", opts, key=f"sel_{i}")
            with c3:
                st.session_state.tracks[i]['durata'] = st.number_input("Durată (sec):", 5, 600, 60, key=f"dur_{i}")

    # --- BUTON GENERARE CU "LIPIRE" (GLUE) ---
    if st.button("🚀 GENEREAZĂ MIXUL PROFESIONAL", type="primary"):
        with st.spinner("Se execută lipirea pieselor..."):
            sr = 44100
            final_audio = None
            
            for i, row in enumerate(st.session_state.tracks):
                # Încărcăm piesa
                y, _ = librosa.load(row['path'], sr=sr, mono=True, duration=row['durata'])
                y = librosa.util.normalize(y)

                # Aplicăm Redrum la finalul piesei dacă e selectat
                if row['drum_loop'] != "Fără":
                    y_d, _ = librosa.load(st.session_state.drum_lib[row['drum_loop']], sr=sr, mono=True)
                    ov_len = min(len(y_d), int(10 * sr), len(y))
                    y[-ov_len:] = (y[-ov_len:] * 0.4) + (y_d[:ov_len] * 0.6)
                
                if final_audio is None:
                    final_audio = y
                else:
                    # MIXAJUL PRO (Lipirea):
                    # Piesa 2 intră PESTE finalul piesei 1 (unde bat tobele)
                    overlap_sec = 5 
                    ov_samples = int(overlap_sec * sr)
                    
                    # Fade-in pe piesa care intră
                    fade_in = np.linspace(0, 1, ov_samples)
                    y_start = y[:ov_samples] * fade_in
                    
                    # Îmbinare
                    mixed_zone = final_audio[-ov_samples:] + y_start
                    final_audio = np.concatenate([final_audio[:-ov_samples], mixed_zone, y[ov_samples:]])

            sf.write("mix_final_pro.wav", final_audio, sr)
            st.audio("mix_final_pro.wav")
            st.download_button("💾 DESCARCĂ", open("mix_final_pro.wav", "rb"), "SmartMix_Pro.wav")
