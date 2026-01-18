import streamlit as st
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import subprocess
import os

st.set_page_config(page_title="DJCIOKO MIXER", layout="wide")

st.title("🎧 DJCIOKO - TikTok Video Mixer")

# --- PASUL 1: ÎNCĂRCARE ---
st.header("1. Încarcă melodiile și poza")
uploaded_files = st.file_uploader("Alege cele 10 melodii (MP3/WAV)", type=['mp3', 'wav'], accept_multiple_files=True)
foto = st.file_uploader("Alege poza pentru fundal", type=['jpg', 'png', 'jpeg'])

# Creăm un spațiu pentru date în memorie
if 'data_finala' not in st.session_state:
    st.session_state.data_finala = None

# --- PASUL 2: ANALIZĂ (BUTONUL ALBASTRU) ---
if uploaded_files:
    if st.button("🔍 ANALIZEAZĂ PIESELE"):
        all_results = []
        p_bar = st.progress(0)
        
        for i, f in enumerate(uploaded_files):
            # Analiză rapidă
            y, sr = librosa.load(f, sr=22050) # am scăzut SR pentru viteză
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Forțăm startul la refren (sau secunda 30 dacă e piesa lungă)
            start_refren = 30 if len(y) > (30 * sr) else 0
            
            all_results.append({
                "Nume": f.name,
                "BPM": round(float(tempo), 1),
                "Start": start_refren,
                "file_obj": f
            })
            p_bar.progress((i + 1) / len(uploaded_files))
        
        # Sortăm după BPM
        st.session_state.data_finala = sorted(all_results, key=lambda x: x['BPM'])
        st.success("✅ Analiza gata! Vezi ordinea mai jos.")

# --- PASUL 3: AFIȘARE ȘI MIXARE ---
if st.session_state.data_finala:
    df = pd.DataFrame(st.session_state.data_finala)[["Nume", "BPM"]]
    st.table(df)
    
    if st.button("🚀 GENEREAZĂ MIX & VIDEO"):
        with st.spinner("Se lucrează la mix..."):
            mix_audio = []
            sr_mix = 44100
            
            for piesa in st.session_state.data_finala:
                # Încărcăm exact 30 secunde
                y, sr = librosa.load(piesa['file_obj'], sr=sr_mix, offset=piesa['Start'], duration=30)
                
                # Fade in/out 0.5s să sune bine
                fade = int(0.5 * sr_mix)
                if len(y) > fade:
                    y[:fade] *= np.linspace(0, 1, fade)
                    y[-fade:] *= np.linspace(1, 0, fade)
                
                mix_audio.extend(y)
            
            # Salvare Audio
            audio_path = "mix.mp3"
            sf.write(audio_path, np.array(mix_audio), sr_mix)
            
            if foto:
                # Salvare Foto
                with open("img.jpg", "wb") as f:
                    f.write(foto.getbuffer())
                
                # Mixare Video cu FFmpeg
                video_path = "DJCIOKO_FINAL.mp4"
                subprocess.run(f"ffmpeg -y -loop 1 -i img.jpg -i {audio_path} -c:v libx264 -tune stillimage -c:a aac -b:a 192k -pix_fmt yuv420p -shortest {video_path}", shell=True)
                
                with open(video_path, "rb") as v:
                    st.download_button("⬇️ DESCARCĂ VIDEO (MP4)", v, file_name="DJCIOKO_FINAL.mp4")
            else:
                with open(audio_path, "rb") as a:
                    st.download_button("⬇️ DESCARCĂ AUDIO (MP3)", a, file_name="DJCIOKO_MIX.mp3")
