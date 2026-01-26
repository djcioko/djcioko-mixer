import streamlit as st
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import os

# Configurare pagină și stil
st.set_page_config(page_title="SmartMix Pro Studio", layout="wide")
st.title("🎧 SmartMix Pro - Automated DJ Studio")
st.markdown("### Mixare Inteligentă: Beat-Match, Voice Detection & Manual Timing")

if 'tracks' not in st.session_state:
    st.session_state.tracks = []

# --- 1. ÎNCĂRCARE FIȘIERE ---
files = st.file_uploader("Încărcați piesele audio (MP3/WAV)", type=['mp3', 'wav'], accept_multiple_files=True)

def analyze_audio(path):
    y, sr = librosa.load(path, sr=22050)
    # Detecție BPM și tobe (Beat Tracking)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Detecție automată voce (eliminare intro instrumental)
    intervals = librosa.effects.split(y, top_db=25)
    v_start = intervals[0][0] / sr if len(intervals) > 0 else 0
    
    # Sincronizare pe primul beat după voce
    sync_start = beat_times[beat_times >= v_start][0] if len(beat_times) > 0 else v_start
    
    return round(float(tempo), 1), round(sync_start, 2)

# --- 2. ANALIZĂ ȘI CONFIGURARE ---
if files:
    if st.button("🔍 ANALIZEAZĂ PIESELE"):
        results = []
        valid_files = [f for f in files if not f.name.startswith("._")]
        for f in valid_files:
            path = f.name
            with open(path, "wb") as tmp:
                tmp.write(f.getbuffer())
            try:
                bpm, start_beat = analyze_audio(path)
                results.append({
                    "Piesa": f.name,
                    "BPM": bpm,
                    "Start Beat (s)": start_beat,
                    "Durata (sec)": 75,  # Default 1:15
                    "file_path": path
                })
            except: continue
        st.session_state.tracks = results
        st.success("Analiză completă! Acum poți ajusta duratele mai jos.")

# --- 3. TABEL EDITABIL (Setare manuală timp) ---
if st.session_state.tracks:
    st.markdown("#### ⚙️ Configurează Durata și Ordinea (Sortat după BPM automat)")
    df = pd.DataFrame(st.session_state.tracks).sort_values(by="BPM")
    
    # Permitem editarea coloanei de durată direct în tabel
    edited_df = st.data_editor(
        df[["Piesa", "BPM", "Start Beat (s)", "Durata (sec)"]],
        column_config={
            "Durata (sec)": st.column_config.NumberColumn(
                "Durata (sec)",
                help="Introdu manual secundele (ex: 60 pentru 1 min, 90 pentru 1:30, 120 pentru 2 min)",
                min_value=10,
                max_value=600,
                step=1,
            )
        },
        disabled=["Piesa", "BPM", "Start Beat (s)"],
        hide_index=True,
    )
    
    # Actualizăm session_state cu valorile noi
    for index, row in edited_df.iterrows():
        for track in st.session_state.tracks:
            if track["Piesa"] == row["Piesa"]:
                track["Durata (sec)"] = row["Durata (sec)"]

    st.markdown("---")
    
    # --- 4. OPȚIUNI EXPORT ---
    col1, col2 = st.columns(2)
    with col1:
        format_ales = st.selectbox("Alege formatul final:", ["MP3 (320 kbps)", "WAV (High Fidelity)"])
    with col2:
        crossfade_val = st.slider("Durată Crossfade (secunde):", 1, 10, 5)

    if st.button("🚀 GENEREAZĂ MIXUL FINAL"):
        with st.spinner("SmartMix Pro procesează mixul tău pe beat..."):
            sr_mix = 44100
            final_mix = np.array([], dtype=np.float32)
            sorted_tracks = sorted(st.session_state.tracks, key=lambda x: x['BPM'])
            
            for i, t in enumerate(sorted_tracks):
                # Încărcare cu offset (start pe beat) și durată manuală
                y, _ = librosa.load(t['file_path'], sr=sr_mix, offset=t['Start Beat (s)'], duration=t['Durata (sec)'])
                
                # Uniformizare volum (RMS)
                rms = np.sqrt(np.mean(y**2))
                if rms > 0: y = y * (0.12 / rms)
                
                fade_samples = int(crossfade_val * sr_mix)
                
                if i == 0:
                    final_mix = y
                else:
                    # Sincronizare crossfade
                    out_part = final_mix[-fade_samples:] * np.linspace(1, 0, fade_samples)
                    in_part = y[:fade_samples] * np.linspace(0, 1, fade_samples)
                    final_mix[-fade_samples:] = out_part + in_part
                    final_mix = np.concatenate([final_mix, y[fade_samples:]])
            
            final_mix = np.clip(final_mix, -1, 1)
            
            # Export în funcție de alegere
            if "MP3" in format_ales:
                temp_wav = "temp_mix.wav"
                sf.write(temp_wav, final_mix, sr_mix, subtype='PCM_24')
                iesire = "SmartMix_Pro_Master.mp3"
                os.system(f"ffmpeg -i {temp_wav} -ab 320k -y {iesire}")
                ext = "mp3"
            else:
                iesire = "SmartMix_Pro_Master.wav"
                sf.write(iesire, final_mix, sr_mix, subtype='PCM_24')
                ext = "wav"
            
            with open(iesire, "rb") as f_out:
                st.download_button(f"⬇️ DESCARCĂ MIX {format_ales}", f_out, file_name=iesire)

st.markdown("---")
st.caption("© 2026 SmartMix Pro | Beat-Matching AI Engine")
