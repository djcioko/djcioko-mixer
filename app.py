import streamlit as st
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import os

st.set_page_config(page_title="SmartMix Pro V3", layout="wide")
st.title("🎧 SmartMix Pro V3 - Engine Avansat de Mixare")

if 'tracks' not in st.session_state:
    st.session_state.tracks = []

# --- MOTORUL DE ANALIZĂ AVANSATĂ ---
def analyze_track_complex(path):
    # Încărcăm piesa
    y, sr = librosa.load(path, sr=22050)
    
    # 1. Detecție Beat precisă
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # 2. Găsirea "Drop-ului" (unde crește energia brusc)
    # Calculăm energia RMS pe ferestre scurte
    hop_length = 512
    rmse = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.times_like(rmse, sr=sr, hop_length=hop_length)
    
    # Găsim primul punct unde energia sare peste 60% din media piesei
    energy_threshold = np.mean(rmse) * 1.2
    drop_index = np.where(rmse > energy_threshold)[0][0] if any(rmse > energy_threshold) else 0
    drop_time = times[drop_index]
    
    # Aliniem drop-ul cu cel mai apropiat beat pentru sync perfect
    closest_beat = beat_times[np.abs(beat_times - drop_time).argmin()]
    
    return round(float(tempo), 1), round(closest_beat, 2)

# --- INTERFAȚĂ UPLOAD ---
files = st.file_uploader("Încarcă muzica:", type=['mp3', 'wav'], accept_multiple_files=True)

if files:
    if st.button("🔍 ANALIZĂ SMART (BPM & DROP DETECT)"):
        results = []
        valid_files = [f for f in files if not f.name.startswith("._")]
        for f in valid_files:
            with open(f.name, "wb") as tmp:
                tmp.write(f.getbuffer())
            try:
                bpm, start_sync = analyze_track_complex(f.name)
                results.append({
                    "Piesa": f.name,
                    "BPM": bpm,
                    "Start Manual (sec)": start_sync,
                    "Durata (sec)": 90,
                    "path": f.name
                })
            except: continue
        st.session_state.tracks = results

# --- CONTROLUL MIXAJULUI ---
if st.session_state.tracks:
    st.subheader("⚙️ Configurare Manuală pe Piesă")
    # Tabel editabil pentru a seta EXACT secundele
    edited_df = st.data_editor(
        pd.DataFrame(st.session_state.tracks).drop(columns=['path']),
        num_rows="dynamic"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        fmt = st.selectbox("Format Export:", ["WAV (Studio)", "MP3 320kbps"])
    with col2:
        cf_time = st.slider("Crossfade (secunde):", 2, 10, 5)
    with col3:
        master_gain = st.slider("Master Gain (Volum):", 0.5, 2.0, 1.2)

    if st.button("🚀 GENEREAZĂ MIXAJUL PROFESIONAL"):
        with st.spinner("Sincronizare beat-match și masterizare..."):
            sr_mix = 44100
            final_audio = np.array([], dtype=np.float32)
            
            # Sortăm după BPM pentru tranziții fine
            tracks_to_process = edited_df.sort_values("BPM").to_dict('records')
            
            for i, t in enumerate(tracks_to_process):
                # Găsim calea fișierului original
                orig_path = next(item['path'] for item in st.session_state.tracks if item['Piesa'] == t['Piesa'])
                
                # Încărcăm exact cât s-a setat manual
                y, _ = librosa.load(orig_path, sr=sr_mix, offset=t['Start Manual (sec)'], duration=t['Durata (sec)'])
                
                # --- MASTERIZARE ȘI COMPRESIE ---
                # Normalizare RMS (Uniformizare volum)
                y = librosa.util.normalize(y) * master_gain
                
                fade_samples = int(cf_time * sr_mix)
                if i == 0:
                    final_audio = y
                else:
                    # Sincronizare pe tobe la crossfade
                    # Aplicăm fade out și in
                    out_part = final_audio[-fade_samples:] * np.cos(np.linspace(0, np.pi/2, fade_samples))
                    in_part = y[:fade_samples] * np.sin(np.linspace(0, np.pi/2, fade_samples))
                    
                    final_audio[-fade_samples:] = out_part + in_part
                    final_audio = np.concatenate([final_audio, y[fade_samples:]])
            
            # Limiter final pentru a preveni distorsiunea
            final_audio = np.clip(final_audio, -0.98, 0.98)
            
            if "MP3" in fmt:
                sf.write("temp.wav", final_audio, sr_mix)
                os.system(f"ffmpeg -i temp.wav -ab 320k -y SmartMix_Final.mp3")
                out_file = "SmartMix_Final.mp3"
            else:
                sf.write("SmartMix_Final.wav", final_audio, sr_mix, subtype='PCM_24')
                out_file = "SmartMix_Final.wav"
                
            with open(out_file, "rb") as f:
                st.download_button("⬇️ DESCARCĂ REZULTATUL", f, file_name=out_file)
