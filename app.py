import streamlit as st
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import os

st.set_page_config(page_title="SmartMix Pro V3.4", layout="wide")
st.title("🎧 SmartMix Pro - Control Total Ordine & BPM")

if 'tracks' not in st.session_state:
    st.session_state.tracks = []

def get_pro_start(path):
    y, sr = librosa.load(path, sr=22050, duration=60)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)[0]
    start_s = librosa.frames_to_time(np.where(rmse > np.mean(rmse)*1.3)[0][0]) if any(rmse > np.mean(rmse)*1.3) else 0
    return round(float(tempo), 1), round(float(start_s), 2)

files = st.file_uploader("Încarcă muzica:", type=['mp3', 'wav'], accept_multiple_files=True)

if files:
    if st.button("🔍 ANALIZEAZĂ PIESELE"):
        results = []
        for f in files:
            if f.name.startswith("._"): continue
            with open(f.name, "wb") as tmp:
                tmp.write(f.getbuffer())
            try:
                bpm, s_start = get_pro_start(f.name)
                results.append({"Piesa": f.name, "BPM": bpm, "Start (sec)": s_start, "Durata (sec)": 90.0, "path": f.name})
            except:
                results.append({"Piesa": f.name, "BPM": 120.0, "Start (sec)": 0.0, "Durata (sec)": 90.0, "path": f.name})
        st.session_state.tracks = results

if st.session_state.tracks:
    st.markdown("### ⚙️ Aranjare Ordine & Sortare BPM")
    
    # Butoane pentru sortare automată
    col_s1, col_s2, col_s3 = st.columns([1, 1, 4])
    df_temp = pd.DataFrame(st.session_state.tracks)
    
    if col_s1.button("⬅️ BPM Mic la Mare"):
        st.session_state.tracks = df_temp.sort_values(by="BPM", ascending=True).to_dict('records')
    if col_s2.button("➡️ BPM Mare la Mic"):
        st.session_state.tracks = df_temp.sort_values(by="BPM", ascending=False).to_dict('records')

    # Tabel editabil cu formatare forțată pentru a elimina simbolurile ciudate
    df_disp = pd.DataFrame(st.session_state.tracks).drop(columns=['path'])
    edited_df = st.data_editor(
        df_disp, 
        num_rows="dynamic", 
        key="pro_editor",
        column_config={
            "BPM": st.column_config.NumberColumn("BPM", format="%.1f"), # Elimină simbolurile %, $ sau €
            "Start (sec)": st.column_config.NumberColumn("Start (sec)", format="%.2f"),
            "Durata (sec)": st.column_config.NumberColumn("Durata (sec)", format="%d")
        }
    )

    col1, col2 = st.columns(2)
    with col1:
        fmt = st.selectbox("Format:", ["WAV", "MP3 320kbps"])
    with col2:
        cf = st.slider("Crossfade (secunde):", 2, 10, 5)

    if st.button("🚀 GENEREAZĂ MIXUL"):
        try:
            with st.spinner("Se creează mixul în ordinea stabilită..."):
                sr_mix = 44100
                final = np.array([], dtype=np.float32)
                
                # Urmărim EXACT ordinea din tabelul editat
                for i, row in edited_df.iterrows():
                    p_path = next(item['path'] for item in st.session_state.tracks if item['Piesa'] == row['Piesa'])
                    y, _ = librosa.load(p_path, sr=sr_mix, offset=float(row['Start (sec)']), duration=float(row['Durata (sec)']))
                    y = librosa.util.normalize(y) * 0.95
                    f_len = int(cf * sr_mix)
                    if i == 0:
                        final = y
                    else:
                        o_f = final[-f_len:] * np.linspace(1, 0, f_len)
                        i_f = y[:f_len] * np.linspace(0, 1, f_len)
                        final[-f_len:] = o_f + i_f
                        final = np.concatenate([final, y[f_len:]])
                
                out_name = f"SmartMix_Pro_Custom.{'mp3' if 'MP3' in fmt else 'wav'}"
                if "MP3" in fmt:
                    sf.write("t.wav", final, sr_mix)
                    os.system(f"ffmpeg -i t.wav -ab 320k -y {out_name}")
                else:
                    sf.write(out_name, final, sr_mix, subtype='PCM_24')
                st.audio(out_name)
                with open(out_name, "rb") as f_res:
                    st.download_button("⬇️ DESCARCĂ REZULTATUL", f_res, file_name=out_name)
        except Exception as e:
            st.error(f"Eroare: {e}")
