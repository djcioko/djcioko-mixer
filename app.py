import streamlit as st
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import os

st.set_page_config(page_title="DJCIOKO SMART MIXER", layout="wide")
st.title("🎧 DJCIOKO - AUTO CUT (Voice Start & Smart Transition)")

if 'tracks' not in st.session_state:
    st.session_state.tracks = []

# --- 1. UPLOAD ---
files = st.file_uploader("Încarcă melodiile (MP3/WAV)", type=['mp3', 'wav'], accept_multiple_files=True)

# --- FUNCTIE DETECTIE VOCE ---
def get_voice_start(y, sr, top_db=25):
    # Detectează zonele cu semnal sonor peste pragul de liniște
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) > 0:
        return intervals[0][0]
    return 0

# --- 2. ANALIZĂ ---
if files:
    if st.button("🔍 ANALIZEAZĂ VOCE ȘI BPM"):
        results = []
        valid_files = [f for f in files if not f.name.startswith("._")]
        status = st.empty()
        
        for f in valid_files:
            status.text(f"Se procesează: {f.name}...")
            path = f.name
            with open(path, "wb") as tmp:
                tmp.write(f.getbuffer())
            
            try:
                # Citire pentru analiză
                y_an, sr_an = librosa.load(path, sr=22050, duration=90)
                
                # Detectare start voce
                start_sample = get_voice_start(y_an, sr_an)
                start_sec = start_sample / sr_an
                
                # BPM
                tempo, _ = librosa.beat.beat_track(y=y_an[start_sample:], sr=sr_an)
                
                # Durată variabilă (între 1:15 și 2:00 minute)
                duration = 120 if tempo < 115 else 75
                
                results.append({
                    "Melodie": f.name,
                    "BPM": round(float(tempo), 1),
                    "Start Sec": round(start_sec, 2),
                    "Durata Mix": duration,
                    "file_path": path
                })
            except:
                continue
        
        st.session_state.tracks = sorted(results, key=lambda x: x['BPM'])
        status.success("✅ Analiză gata!")

# --- 3. AFIȘARE ȘI MIXARE ---
if st.session_state.tracks:
    # Verificăm dacă datele sunt corecte înainte de tabel pentru a evita KeyError
    df = pd.DataFrame(st.session_state.tracks)
    cols_to_show = ["Melodie", "BPM", "Start Sec", "Durata Mix"]
    st.table(df[cols_to_show])
    
    if st.button("🚀 GENEREAZĂ MIXUL CU VOICE START"):
        with st.spinner("Se uniformizează volumul și se aplică crossfade de 5s..."):
            sr_mix = 44100
            fade_sec = 5 
            final_mix = np.array([], dtype=np.float32)
            
            for i, t in enumerate(st.session_state.tracks):
                # Încărcare segment fix pornind de la voce
                y, _ = librosa.load(t['file_path'], sr=sr_mix, offset=t['Start Sec'], duration=t['Durata Mix'])
                
                # --- UNIFORMIZARE VOLUM (Loudness Normalization) ---
                rms = np.sqrt(np.mean(y**2))
                if rms > 0:
                    y = y * (0.12 / rms) # Aduce toate piesele la același nivel mediu
                
                fade_samples = int(fade_sec * sr_mix)
                if i == 0:
                    final_mix = y
                else:
                    # Crossfade profesional
                    out_part = final_mix[-fade_samples:] * np.linspace(1, 0, fade_samples)
                    in_part = y[:fade_samples] * np.linspace(0, 1, fade_samples)
                    final_mix[-fade_samples:] = out_part + in_part
                    final_mix = np.concatenate([final_mix, y[fade_samples:]])
            
            # Limitare vârfuri pentru a evita distorsiunea
            final_mix = np.clip(final_mix, -1, 1)
            
            iesire = "DJCIOKO_PRO_MIX.mp3"
            sf.write(iesire, final_mix, sr_mix)
            
            with open(iesire, "rb") as f_out:
                st.download_button("⬇️ DESCARCĂ MIXUL FINAL", f_out, file_name=iesire)
