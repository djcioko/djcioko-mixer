import streamlit as st
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import os

st.set_page_config(page_title="DJCIOKO AUTO CUT", layout="wide")
st.title("🎧 DJCIOKO - AUTO CUT MIX DJ (30s + Normalizare)")

if 'tracks' not in st.session_state:
    st.session_state.tracks = []

# --- 1. UPLOAD ---
files = st.file_uploader("Încarcă melodiile (MP3/WAV)", type=['mp3', 'wav'], accept_multiple_files=True)

# --- 2. ANALIZĂ ---
if files:
    if st.button("🔍 ANALIZEAZĂ ȘI SORTEAZĂ BPM"):
        results = []
        status = st.empty()
        for f in files:
            status.text(f"Se procesează: {f.name}...")
            # Salvăm fișierul local pentru ca librosa să îl găsească ușor
            with open(f.name, "wb") as tmp:
                tmp.write(f.getbuffer())
            
            try:
                # Citim cu sr=None pentru a păstra calitatea nativă la analiză
                y, sr = librosa.load(f.name, sr=22050, duration=45)
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                
                results.append({
                    "Melodie": f.name,
                    "BPM": round(float(tempo), 1),
                    "file_path": f.name
                })
            except Exception as e:
                st.error(f"Eroare la fișierul {f.name}. Asigură-te că e un MP3 valid.")
        
        st.session_state.tracks = sorted(results, key=lambda x: x['BPM'])
        status.success("✅ Analiză gata! Acum poți genera mixul.")

# --- 3. MIXARE ---
if st.session_state.tracks:
    st.table(pd.DataFrame(st.session_state.tracks)[["Melodie", "BPM"]])
    
    if st.button("🚀 GENEREAZĂ MIXUL PROFESIONAL"):
        with st.spinner("Se uniformizează volumul și se aplică crossfade..."):
            sr_mix = 44100
            fade_sec = 2 
            
            final_mix = np.array([], dtype=np.float32)
            
            for i, t in enumerate(st.session_state.tracks):
                y, _ = librosa.load(t['file_path'], sr=sr_mix, duration=30)
                
                # Normalizare volum (Uniformizare)
                rms = np.sqrt(np.mean(y**2))
                if rms > 0:
                    y = y * (0.15 / rms)
                
                # Logică simplă de crossfade
                fade_samples = int(fade_sec * sr_mix)
                if i == 0:
                    final_mix = y
                else:
                    # Suprapunere cu fade
                    overlap = final_mix[-fade_samples:] * np.linspace(1, 0, fade_samples)
                    start_new = y[:fade_samples] * np.linspace(0, 1, fade_samples)
                    final_mix[-fade_samples:] = overlap + start_new
                    final_mix = np.concatenate([final_mix, y[fade_samples:]])
            
            iesire = "DJCIOKO_NOSTALGIA_MIX.mp3"
            sf.write(iesire, final_mix, sr_mix)
            
            with open(iesire, "rb") as f_out:
                st.download_button("⬇️ DESCARCĂ MIXUL", f_out, file_name=iesire)
