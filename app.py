import streamlit as st
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import os

st.set_page_config(page_title="SmartMix Pro V3.6", layout="wide")
st.title("🎧 SmartMix Pro - Expert Manual Control")

if 'tracks' not in st.session_state:
    st.session_state.tracks = []

def get_pro_start(path):
    y, sr = librosa.load(path, sr=22050, duration=60)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)[0]
    start_s = librosa.frames_to_time(np.where(rmse > np.mean(rmse)*1.3)[0][0]) if any(rmse > np.mean(rmse)*1.3) else 0
    return round(float(tempo), 1), round(float(start_s), 2)

# --- SIDEBAR CONFIG ---
st.sidebar.header("🚀 Setări Mix")
durata_globala = st.sidebar.slider("Durată piese (sec):", 30, 600, 120)
cf_global = st.sidebar.slider("Crossfade (sec):", 2, 15, 5)
format_export = st.sidebar.selectbox("Format:", ["WAV", "MP3 320kbps"])

files = st.file_uploader("Încarcă muzica:", type=['mp3', 'wav'], accept_multiple_files=True)

if files:
    if st.button("🔍 ANALIZEAZĂ ȘI ADAUGĂ"):
        results = []
        for f in files:
            if f.name.startswith("._"): continue
            with open(f.name, "wb") as tmp:
                tmp.write(f.getbuffer())
            try:
                bpm, s_start = get_pro_start(f.name)
                # Adăugăm un index de ordine (poziție)
                results.append({
                    "Ordine": len(st.session_state.tracks) + len(results) + 1,
                    "Piesa": f.name, 
                    "BPM": bpm, 
                    "Start (sec)": s_start, 
                    "Durata (sec)": float(durata_globala),
                    "path": f.name
                })
            except:
                results.append({"Ordine": 99, "Piesa": f.name, "BPM": 120.0, "Start (sec)": 0.0, "Durata (sec)": float(durata_globala), "path": f.name})
        st.session_state.tracks.extend(results)

if st.session_state.tracks:
    st.markdown("### ⚙️ Tabel Editabil: Schimbă 'Ordine' pentru a rearanja manual")
    
    # Afișăm tabelul editabil
    df_disp = pd.DataFrame(st.session_state.tracks).drop(columns=['path'])
    
    edited_df = st.data_editor(
        df_disp, 
        num_rows="dynamic", 
        key="v36_editor",
        column_config={
            "Ordine": st.column_config.NumberColumn("Poz.", format="%d", help="Schimbă numărul pentru a muta piesa în mix"),
            "BPM": st.column_config.NumberColumn("BPM", format="%.1f"),
            "Start (sec)": st.column_config.NumberColumn("Start (sec)", format="%.2f"),
            "Durata (sec)": st.column_config.NumberColumn("Durata (sec)", format="%d")
        }
    )

    # Re-sortăm automat session_state bazat pe ce a scris utilizatorul în coloana "Ordine"
    if st.button("🔄 ACTUALIZEAZĂ ORDINEA"):
        st.session_state.tracks = []
        # Reconstruim lista bazată pe noul tabel editat
        for _, row in edited_df.sort_values(by="Ordine").iterrows():
            # Căutăm path-ul în fișierele de pe disc (după nume)
            st.session_state.tracks.append({
                "Ordine": row["Ordine"],
                "Piesa": row["Piesa"],
                "BPM": row["BPM"],
                "Start (sec)": row["Start (sec)"],
                "Durata (sec)": row["Durata (sec)"],
                "path": row["Piesa"] 
            })
        st.success("Ordinea a fost salvată! Mixul va respecta numerele de la 1 la X.")
        st.rerun()

    if st.button("🚀 GENEREAZĂ MIXUL FINAL"):
        try:
            with st.spinner("Mixare în progres conform ordinii tale manuale..."):
                sr_mix = 44100
                final = np.array([], dtype=np.float32)
                
                # Sortăm tabelul după coloana 'Ordine' înainte de procesare
                final_queue = edited_df.sort_values(by="Ordine")
                
                for i, row in final_queue.iterrows():
                    y, _ = librosa.load(row['Piesa'], sr=sr_mix, offset=float(row['Start (sec)']), duration=float(row['Durata (sec)']))
                    y = librosa.util.normalize(y) * 0.95
                    
                    f_samples = int(cf_global * sr_mix)
                    if i == 0 or len(final) == 0:
                        final = y
                    else:
                        out_fade = final[-f_samples:] * np.linspace(1, 0, f_samples)
                        in_fade = y[:f_samples] * np.linspace(0, 1, f_samples)
                        final[-f_samples:] = out_fade + in_fade
                        final = np.concatenate([final, y[f_samples:]])
                
                ext = 'mp3' if 'MP3' in format_export else 'wav'
                out_name = f"SmartMix_Custom_Order.{ext}"
                
                if ext == 'mp3':
                    sf.write("t.wav", final, sr_mix)
                    os.system(f"ffmpeg -i t.wav -ab 320k -y {out_name}")
                else:
                    sf.write(out_name, final, sr_mix, subtype='PCM_24')
                
                st.audio(out_name)
                with open(out_name, "rb") as f_res:
                    st.download_button("⬇️ DESCARCĂ", f_res, file_name=out_name)
        except Exception as e:
            st.error(f"Eroare: {e}")
