import streamlit as st
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import os

st.set_page_config(page_title="SmartMix Pro V3.7", layout="wide")
st.title("🎧 SmartMix Pro - Manual Arrangement Studio")

if 'tracks' not in st.session_state:
    st.session_state.tracks = []

# --- ANALIZĂ ---
def analyze_track(path):
    y, sr = librosa.load(path, sr=22050, duration=45)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)[0]
    start_s = librosa.frames_to_time(np.where(rmse > np.mean(rmse)*1.3)[0][0]) if any(rmse > np.mean(rmse)*1.3) else 0
    return round(float(tempo), 1), round(float(start_s), 2)

# --- SIDEBAR ---
st.sidebar.header("🚀 Setări Mix")
durata_def = st.sidebar.number_input("Durată standard piese (sec):", 30, 600, 120)
cf_sec = st.sidebar.slider("Crossfade (sec):", 2, 15, 5)
fmt_out = st.sidebar.selectbox("Format:", ["WAV", "MP3 320kbps"])

# --- UPLOAD ---
files = st.file_uploader("Încarcă muzica:", type=['mp3', 'wav'], accept_multiple_files=True)

if files:
    if st.button("🔍 ANALIZEAZĂ PIESELE"):
        for f in files:
            if any(t['Piesa'] == f.name for t in st.session_state.tracks): continue
            with open(f.name, "wb") as tmp:
                tmp.write(f.getbuffer())
            try:
                bpm, s_start = analyze_track(f.name)
                st.session_state.tracks.append({
                    "Piesa": f.name, "BPM": bpm, "Start (sec)": s_start, "Durata (sec)": float(durata_def)
                })
            except:
                st.session_state.tracks.append({
                    "Piesa": f.name, "BPM": 120.0, "Start (sec)": 0.0, "Durata (sec)": float(durata_def)
                })
        st.rerun()

# --- GESTIONARE ORDINE ---
if st.session_state.tracks:
    st.markdown("### 📋 Lista de Mixaj (Aranjează Ordinea)")
    
    # Butoane Sortare BPM
    c1, c2, c3 = st.columns([1.5, 1.5, 4])
    if c1.button("📉 Sortează BPM (Mic-Mare)"):
        st.session_state.tracks = sorted(st.session_state.tracks, key=lambda x: x['BPM'])
        st.rerun()
    if c2.button("📈 Sortează BPM (Mare-Mic)"):
        st.session_state.tracks = sorted(st.session_state.tracks, key=lambda x: x['BPM'], reverse=True)
        st.rerun()
    if c3.button("🗑️ Golește Lista"):
        st.session_state.tracks = []
        st.rerun()

    # Afișare rând cu rând cu butoane de mutare
    new_order = []
    for i, track in enumerate(st.session_state.tracks):
        col_move, col_info, col_edit = st.columns([1, 4, 3])
        
        # Butoane Mutare
        with col_move:
            btn_up = st.button("🔼", key=f"up_{i}")
            btn_down = st.button("🔽", key=f"down_{i}")
            
            if btn_up and i > 0:
                st.session_state.tracks[i], st.session_state.tracks[i-1] = st.session_state.tracks[i-1], st.session_state.tracks[i]
                st.rerun()
            if btn_down and i < len(st.session_state.tracks) - 1:
                st.session_state.tracks[i], st.session_state.tracks[i+1] = st.session_state.tracks[i+1], st.session_state.tracks[i]
                st.rerun()
        
        # Informații
        with col_info:
            st.markdown(f"**{i+1}. {track['Piesa']}**")
            st.caption(f"BPM: {track['BPM']} | Start sugerat: {track['Start (sec)']}s")
            
        # Editare Manuală
        with col_edit:
            new_start = st.number_input("Start (s)", value=track['Start (sec)'], key=f"s_{i}", step=0.1)
            new_dur = st.number_input("Durată (s)", value=track['Durata (sec)'], key=f"d_{i}", step=1.0)
            st.session_state.tracks[i]['Start (sec)'] = new_start
            st.session_state.tracks[i]['Durata (sec)'] = new_dur

    st.markdown("---")
    if st.button("🚀 GENEREAZĂ MIXUL FINAL"):
        try:
            with st.spinner("Mixare profesională în curs..."):
                sr_mix = 44100
                final = np.array([], dtype=np.float32)
                
                for i, row in enumerate(st.session_state.tracks):
                    y, _ = librosa.load(row['Piesa'], sr=sr_mix, offset=float(row['Start (sec)']), duration=float(row['Durata (sec)']))
                    y = librosa.util.normalize(y) * 0.95
                    f_len = int(cf_sec * sr_mix)
                    
                    if i == 0:
                        final = y
                    else:
                        out_f = final[-f_len:] * np.linspace(1, 0, f_len)
                        in_f = y[:f_len] * np.linspace(0, 1, f_len)
                        final[-f_len:] = out_f + in_f
                        final = np.concatenate([final, y[f_len:]])
                
                out_name = f"SmartMix_Result.{'mp3' if 'MP3' in fmt_out else 'wav'}"
                if "MP3" in fmt_out:
                    sf.write("t.wav", final, sr_mix)
                    os.system(f"ffmpeg -i t.wav -ab 320k -y {out_name}")
                else:
                    sf.write(out_name, final, sr_mix, subtype='PCM_24')
                
                st.audio(out_name)
                with open(out_name, "rb") as f_res:
                    st.download_button("⬇️ DESCARCĂ MIXUL", f_res, file_name=out_name)
        except Exception as e:
            st.error(f"Eroare: {e}")
