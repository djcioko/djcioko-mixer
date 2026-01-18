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

if 'data_finala' not in st.session_state:
    st.session_state.data_finala = None

# --- PASUL 2: ANALIZĂ ---
if uploaded_files:
    if st.button("🔍 PASUL 1: ANALIZEAZĂ PIESELE"):
        all_results = []
        p_bar = st.progress(0)
        
        # Salvăm fișierele local pentru a evita eroarea Libsndfile
        for i, f in enumerate(uploaded_files):
            with open(f.name, "wb") as temp_f:
                temp_f.write(f.getbuffer())
            
            y, sr = librosa.load(f.name, sr=22050)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            start_refren = 30 if len(y) > (30 * sr) else 0
            
            all_results.append({
                "Nume": f.name,
                "BPM": round(float(tempo), 1),
                "Start": start_refren
            })
            p_bar.progress((i + 1) / len(uploaded_files))
        
        st.session_state.data_finala = sorted(all_results, key=lambda x: x['BPM'])
        st.success("✅ Analiza gata! Vezi ordinea mai jos.")

# --- PASUL 3: AFIȘARE ȘI GENERARE ---
if st.session_state.data_finala:
    df = pd.DataFrame(st.session_state.data_finala)[["Nume", "BPM"]]
    st.table(df)
    
    if st.button("🚀 PASUL 2: GENEREAZĂ MIX & VIDEO"):
        with st.spinner("Se lucrează la mix..."):
            mix_audio = []
            sr_mix = 44100
            
            for piesa in st.session_state.data_finala:
                # Citim de pe disk fișierul salvat la analiza
                y, sr = librosa.load(piesa['Nume'], sr=sr_mix, offset=piesa['Start'], duration=30)
                
                fade = int(0.5 * sr_mix)
                if len(y) > fade:
                    y[:fade] *= np.linspace(0, 1, fade)
                    y[-fade:] *= np.linspace(1, 0, fade)
                
                mix_audio.extend(y)
            
            audio_path = "mix_final.mp3"
            sf.write(audio_path, np.array(mix_audio), sr_mix)
            
            if foto:
                with open("img.jpg", "wb") as f_img:
                    f_img.write(foto.getbuffer())
                
                video_path = "DJCIOKO_FINAL.mp4"
                # FFmpeg comandă forțată
                subprocess.run(f"ffmpeg -y -loop 1 -i img.jpg -i {audio_path} -c:v libx264 -tune stillimage -c:a aac -b:a 192k -pix_fmt yuv420p -shortest {video_path}", shell=True)
                
                with open(video_path, "rb") as v_file:
                    st.download_button("⬇️ PASUL 3: DESCARCĂ VIDEO (MP4)", v_file, file_name="DJCIOKO_FINAL.mp4")
            else:
                with open(audio_path, "rb") as a_file:
                    st.download_button("⬇️ PASUL 3: DESCARCĂ AUDIO (MP3)", a_file, file_name="DJCIOKO_MIX.mp3")
