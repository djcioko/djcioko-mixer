import streamlit as st
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import os

st.set_page_config(page_title="SmartMix Pro", layout="wide")
st.title("🎧 SmartMix Pro - Automated Studio")

if 'tracks' not in st.session_state:
    st.session_state.tracks = []

# --- MOTOR ANALIZĂ AVANSATĂ ---
def analyze_audio_pro(path):
    # Încărcăm piesa pentru analiză (primul minut)
    y, sr = librosa.load(path, sr=22050, duration=60)
    
    # Detecție BPM și Beat Grid
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Detecție Energie (RMS) pentru a găsi "Drop-ul"
    rmse = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rmse, sr=sr)
    energy_threshold = np.mean(rmse) * 1.3
    
    # Găsim momentul unde muzica devine intensă
    drop_idx = np.where(rmse > energy_threshold)[0]
    start_time = times[drop_idx[0]] if len(drop_idx) > 0 else 0
    
    # Aliniem pe cea mai apropiată tobă (Beat Match)
    if len(beat_times) > 0:
        start_time = beat_times[np.abs(beat_times - start_time).argmin()]
        
    return round(float(tempo), 1), round(float(start_time), 2)

# --- ÎNCĂRCARE ---
files = st.file_uploader("Încarcă muzica (MP3/WAV):", type=['mp3', 'wav'], accept_multiple_files=True)

if files:
    if st.button("🔍 ANALIZĂ SMART BEAT"):
        results = []
        for f in files:
            if f.name.startswith("._"): continue
            path = f.name
            with open(path, "wb") as tmp:
                tmp.write(f.getbuffer())
            try:
                bpm, start_s = analyze_audio_pro(path)
                results.append({
                    "Piesa": f.name,
                    "BPM": bpm,
                    "Start (sec)": start_s,
                    "Durata (sec)": 90.0,
                    "path": path
                })
            except Exception as e:
                st.warning(f"Piesa {f.name} necesită setare manuală.")
                results.append({"Piesa": f.name, "BPM": 120.0, "Start (sec)": 0.0, "Durata (sec)": 90.0, "path": path})
        st.session_state.tracks = results

# --- CONTROL ȘI MIXAJ ---
if st.session_state.tracks:
    st.subheader("⚙️ Configurare Mixaj (Poți edita tabelul)")
    # Editare manuală: Timp de start și durată piesa
    df_input = pd.DataFrame(st.session_state.tracks).drop(columns=['path'])
    edited_df = st.data_editor(df_input, num_rows="dynamic")

    col1, col2, col3 = st.columns(3)
    with col1:
        fmt = st.selectbox("Format:", ["WAV", "MP3 320kbps"])
    with col2:
        cf_time = st.slider("Crossfade (sec):", 1, 10, 5)
    with col3:
        master_vol = st.slider("Master Volume:", 0.5, 1.5, 1.0)

    if st.button("🚀 GENEREAZĂ MIXUL PROFESIONAL"):
        with st.spinner("Se execută beat-matching și masterizare..."):
            sr_mix = 44100
            final_audio = np.array([], dtype=np.float32)
            
            # Procesăm piesele în ordinea din tabel
            for i, row in edited_df.iterrows():
                p_info = next(item for item in st.session_state.tracks if item['Piesa'] == row['Piesa'])
                
                # Încărcăm exact segmentul dorit
                y, _ = librosa.load(p_info['path'], sr=sr_mix, 
                                   offset=float(row['Start (sec)']), 
                                   duration=float(row['Durata (sec)']))
                
                # Uniformizare Volum (Normalizare)
                y = librosa.util.normalize(y) * master_vol
                
                fade_len = int(cf_time * sr_mix)
                if i == 0:
                    final_audio = y
                else:
                    # Tranziție Beat-Match (Fade liniar)
                    out_f = final_audio[-fade_len:] * np.linspace(1, 0, fade_len)
                    in_f = y[:fade_len] * np.linspace(0, 1, fade_len)
                    final_audio[-fade_len:] = out_f + in_f
                    final_audio = np.concatenate([final_audio, y[fade_len:]])
            
            # Limiter pentru prevenire distorsiune
            final_audio = np.clip(final_audio, -0.99, 0.99)
            
            out_file = f"SmartMix_Final.{'mp3' if 'MP3' in fmt else 'wav'}"
            if "MP3" in fmt:
                sf.write("t.wav", final_audio, sr_mix)
                os.system(f"ffmpeg -i t.wav -ab 320k -y {out_file}")
            else:
                sf.write(out_file, final_audio, sr_mix, subtype='PCM_24')
            
            with open(out_file, "rb") as f_down:
                st.download_button("⬇️ DESCARCĂ MIXUL MASTERIZAT", f_down, file_name=out_file)

st.markdown("---")
st.caption("© 2026 SmartMix Pro | Professional Beat-Sync Engine")
