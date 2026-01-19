import streamlit as st
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import os

st.set_page_config(page_title="DJCIOKO AUDIO MIXER", layout="wide")
st.title("🎧 DJCIOKO - TikTok Audio Mixer")

# Resetăm fișierele temporare la pornire
if 'tracks' not in st.session_state:
    st.session_state.tracks = []

# --- PASUL 1: ÎNCĂRCARE ---
st.header("1. Încarcă melodiile (MP3/WAV)")
uploaded_files = st.file_uploader("Trage cele 10 melodii aici:", type=['mp3', 'wav'], accept_multiple_files=True)

# --- PASUL 2: ANALIZĂ ȘI SORTARE ---
if uploaded_files:
    if st.button("🔍 ANALIZEAZĂ ȘI SORTEAZĂ DUPĂ BPM"):
        all_results = []
        p_bar = st.progress(0)
        
        for i, f in enumerate(uploaded_files):
            # Salvare fizică pe disk pentru a preveni erorile de citire
            with open(f.name, "wb") as tmp_file:
                tmp_file.write(f.getbuffer())
            
            # Analiză BPM
            y, sr = librosa.load(f.name, sr=22050, duration=60)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Păstrăm informațiile pentru mix
            all_results.append({
                "Nume": f.name,
                "BPM": round(float(tempo), 1),
                "path": f.name
            })
            p_bar.progress((i + 1) / len(uploaded_files))
        
        # Sortare automată: BPM mic -> BPM mare
        st.session_state.tracks = sorted(all_results, key=lambda x: x['BPM'])
        st.success("✅ Analiza este gata!")

# --- PASUL 3: AFIȘARE ȘI DESCĂRCARE ---
if st.session_state.tracks:
    st.write("### Ordinea melodiilor în mix:")
    st.table(pd.DataFrame(st.session_state.tracks)[["Nume", "BPM"]])
    
    if st.button("🚀 GENEREAZĂ MIXUL (MP3)"):
        with st.spinner("Se creează mixul de 30s per piesă..."):
            mix_audio = []
            sr_mix = 44100
            
            for piesa in st.session_state.tracks:
                # Încărcăm fix 30 secunde
                # Începem de la secunda 30 (unde e de obicei refrenul) sau de la 0 dacă e scurtă
                y, _ = librosa.load(piesa['path'], sr=sr_mix, offset=0, duration=30)
                
                # Crossfade scurt să sune bine
                fade = int(0.5 * sr_mix)
                if len(y) > fade:
                    y[:fade] *= np.linspace(0, 1, fade)
                    y[-fade:] *= np.linspace(1, 0, fade)
                
                mix_audio.extend(y)
            
            # Salvare fișier final
            nume_iesire = "DJCIOKO_MIX_FINAL.mp3"
            sf.write(nume_iesire, np.array(mix_audio), sr_mix)
            
            # Butonul de download
            with open(nume_iesire, "rb") as final_file:
                st.download_button(
                    label="⬇️ DESCARCĂ MIXUL DJCIOKO (MP3)",
                    data=final_file,
                    file_name=nume_iesire,
                    mime="audio/mpeg"
                )
