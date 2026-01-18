import streamlit as st
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import os

st.set_page_config(page_title="DJCIOKO MIXER", page_icon="🎧")

st.title("🎧 DJCIOKO - TikTok MegaMixer")
st.markdown("### Mixare automată pe refren & Sortare BPM")

# Reținem fișierele în sesiune (conform cerinței tale)
if 'tracks' not in st.session_state:
    st.session_state.tracks = []

uploaded_files = st.file_uploader("Încarcă melodiile (Bulk Upload)", type=['mp3', 'wav'], accept_multiple_files=True)

if uploaded_files:
    if st.button("🔍 Analizează și Sortează"):
        results = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            # Analiză tehnică
            y, sr = librosa.load(file, duration=120)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Detecție Refren (Peak Energy)
            energy = librosa.feature.rms(y=y)[0]
            duration_30s = int(30 * sr / 512)
            max_idx = np.argmax([np.mean(energy[j:j+duration_30s]) for j in range(len(energy)-duration_30s)])
            start_sec = (max_idx * 512) / sr
            
            results.append({
                "Nume": file.name,
                "BPM": round(float(tempo), 1),
                "Start Refren (s)": round(start_sec, 2),
                "raw_data": y,
                "sr": sr
            })
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Sortare automată BPM crescător
        st.session_state.tracks = sorted(results, key=lambda x: x['BPM'])

if st.session_state.tracks:
    st.write("### ✅ Ordine Mixare (Tempo Scăzut -> Crescător)")
    df = pd.DataFrame(st.session_state.tracks)[["Nume", "BPM", "Start Refren (s)"]]
    st.table(df)

    if st.button("🚀 GENEREAZĂ MIX FINAL"):
        mix_final = []
        sr_mix = 44100
        
        for piesa in st.session_state.tracks:
            # Tăiem 30 de secunde exact de la refren
            start_f = int(piesa['Start Refren (s)'] * sr_mix)
            end_f = start_f + (30 * sr_mix)
            segment = piesa['raw_data'][start_f:end_f]
            
            # Cut Fader (0.1s fade)
            fade = int(0.1 * sr_mix)
            if len(segment) > fade:
                segment[:fade] *= np.linspace(0, 1, fade)
                segment[-fade:] *= np.linspace(1, 0, fade)
            
            mix_final.extend(segment)
        
        # Salvare buffer
        nume_iesire = "DJCIOKO_MIX_TIKTOK.mp3"
        sf.write(nume_iesire, np.array(mix_final), sr_mix)
        
        with open(nume_iesire, "rb") as f:
            st.download_button("⬇️ DESCARCĂ MIXUL", f, file_name=nume_iesire)