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

# --- FUNCTIE DETECTIE VOCE (Taie Intro Instrumental) ---
def get_voice_start(y, sr, top_db=20):
    # detectează zonele care nu sunt tăcere (elimină intro-ul foarte încet sau instrumentalul slab)
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) > 0:
        return intervals[0][0]  # Returnează indexul unde începe sunetul mai tare (primul vers)
    return 0

# --- 2. ANALIZĂ ---
if files:
    if st.button("🔍 ANALIZEAZĂ VOCE ȘI BPM"):
        results = []
        valid_files = [f for f in files if not f.name.startswith("._")]
        
        for f in valid_files:
            with open(f.name, "wb") as tmp:
                tmp.write(f.getbuffer())
            
            try:
                # Încărcăm piesa întreagă pentru analiză
                y, sr = librosa.load(f.name, sr=22050)
                
                # Detectăm startul vocii/versurilor
                start_sample = get_voice_start(y, sr)
                start_sec = start_sample / sr
                
                # Analiză BPM
                tempo, _ = librosa.beat.beat_track(y=y[start_sample:], sr=sr)
                
                # Calculăm o durată variabilă (între 75s și 120s) în funcție de BPM
                # Piesele mai rapide le facem puțin mai scurte, cele lente mai lungi
                duration = 120 if tempo < 110 else 75
                
                results.append({
                    "Melodie": f.name,
                    "BPM": round(float(tempo), 1),
                    "Start Sec": round(start_sec, 2),
                    "Durata Mix": duration,
                    "file_path": f.name
                })
            except Exception:
                continue
        
        st.session_state.tracks = sorted(results, key=lambda x: x['BPM'])
        st.success(f"✅ Analiză completă! Am detectat începutul versurilor pentru {len(st.session_state.tracks)} piese.")

# --- 3. MIXARE PROFESIONALĂ ---
if st.session_state.tracks:
    st.table(pd.DataFrame(st.session_state.tracks)[["Melodie", "BPM", "Start Sec", "Durata Mix"]])
    
    if st.button("🚀 GENEREAZĂ MIXUL CU VOIX START"):
        with st.spinner("Se uniformizează volumul și se aplică crossfade de 5s..."):
            sr_mix = 44100
            fade_sec = 5 
            final_mix = np.array([], dtype=np.float32)
            
            for i, t in enumerate(st.session_state.tracks):
                # Încărcăm piesa pornind fix de la primul vers detectat
                y, _ = librosa.load(t['file_path'], sr=sr_mix, offset=t['Start Sec'], duration=t['Durata Mix'])
                
                # NORMALIZARE VOLUM (Toate piesele la același nivel)
                peak = np.max(np.abs(y))
                if peak > 0:
                    y = y * (0.8 / peak) # Aduce volumul la 80% din maxim constant
                
                fade_samples = int(fade_sec * sr_mix)
                
                if i == 0:
                    final_mix = y
                else:
                    # CROSSFADE 5 SECUNDE
                    # Fade out pe mixul vechi
                    out_part = final_mix[-fade_samples:] * np.linspace(1, 0, fade_samples)
                    # Fade in pe melodia nouă
                    in_part = y[:fade_samples] * np.linspace(0, 1, fade_samples)
                    
                    final_mix[-fade_samples:] = out_part + in_part
                    final_mix = np.concatenate([final_mix, y[fade_samples:]])
            
            iesire = "DJCIOKO_VOICE_MIX.mp3"
            sf.write(iesire, final_mix, sr_mix)
            
            with open(iesire, "rb") as f_out:
                st.download_button("⬇️ DESCARCĂ MIXUL FINAL (SMART CUT)", f_out, file_name=iesire)
