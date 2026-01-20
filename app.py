import streamlit as st
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import os

st.set_page_config(page_title="DJCIOKO AUTO CUT", layout="wide")
st.title("🎧 DJCIOKO - AUTO CUT MIX DJ (30s + Smooth Transitions)")

if 'tracks' not in st.session_state:
    st.session_state.tracks = []

# --- 1. UPLOAD ---
st.subheader("🎵 Pasul 1: Încarcă melodiile")
files = st.file_uploader("Alege piesele:", type=['mp3', 'wav'], accept_multiple_files=True)

# --- 2. ANALIZĂ ---
if files:
    if st.button("🔍 ANALIZEAZĂ ȘI SORTEAZĂ BPM"):
        results = []
        status = st.empty()
        for f in files:
            status.text(f"Se analizează: {f.name}...")
            with open(f.name, "wb") as tmp:
                tmp.write(f.getbuffer())
            
            y, sr = librosa.load(f.name, sr=22050, duration=45)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            results.append({
                "Melodie": f.name,
                "BPM": round(float(tempo), 1),
                "file_path": f.name
            })
        
        st.session_state.tracks = sorted(results, key=lambda x: x['BPM'])
        status.success("✅ Analiză gata! Volumul și tranzițiile sunt pregătite.")

# --- 3. MIXARE CU CROSSFADE ȘI NORMALIZARE ---
if st.session_state.tracks:
    st.table(pd.DataFrame(st.session_state.tracks)[["Melodie", "BPM"]])
    
    if st.button("🚀 GENEREAZĂ MIXUL PROFESIONAL"):
        with st.spinner("Se uniformizează volumul și se aplică crossfade..."):
            sr_mix = 44100
            crossfade_sec = 2 # Durata tranziției în secunde
            segment_duration = 30 # Durata fiecărei piese
            
            final_mix = np.array([], dtype=np.float32)
            
            for i, t in enumerate(st.session_state.tracks):
                # Încărcăm piesa
                y, _ = librosa.load(t['file_path'], sr=sr_mix, duration=segment_duration)
                
                # --- NORMALIZARE VOLUM ---
                # Aduce volumul la un nivel standard de -20dB RMS aproximativ
                rms = np.sqrt(np.mean(y**2))
                if rms > 0:
                    y = y * (0.15 / rms)
                
                # --- LOGICĂ CROSSFADE ---
                fade_samples = int(crossfade_sec * sr_mix)
                
                # Creăm curbele de fade
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                
                if i == 0:
                    # Prima piesă: doar adăugăm
                    final_mix = y
                else:
                    # Suprapunem sfârșitul mixului actual cu începutul piesei noi
                    overlap_start = len(final_mix) - fade_samples
                    
                    # Aplicăm fade-out pe finalul mixului existent
                    final_mix[overlap_start:] *= fade_out
                    
                    # Aplicăm fade-in pe începutul piesei noi
                    new_segment_start = y[:fade_samples] * fade_in
                    
                    # Combinăm (Mixăm) cele două părți
                    final_mix[overlap_start:] += new_segment_start
                    
                    # Adăugăm restul piesei noi (după zona de fade)
                    final_mix = np.concatenate([final_mix, y[fade_samples:]])
            
            # Salvare finală
            iesire = "DJCIOKO_PRO_MIX.mp3"
            sf.write(iesire, final_mix, sr_mix)
            
            with open(iesire, "rb") as final:
                st.download_button("⬇️ DESCARCĂ MIXUL PRO (MP3)", data=final, file_name=iesire)
