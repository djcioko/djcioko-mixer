import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import os
import tempfile

st.set_page_config(page_title="SmartMix Pro V4.0 - Redrum Edition", layout="wide", page_icon="🥁")

# CSS pentru o interfață modernă
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; font-weight: bold; }
    .stSelectbox, .stNumberInput { background-color: #1e2130; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎧 SmartMix Pro V4.0")
st.subheader("Manual Mix & Redrum Transition Studio")

# --- INITIALIZARE SESSION STATE ---
if 'tracks' not in st.session_state:
    st.session_state.tracks = []
if 'drum_slots' not in st.session_state:
    st.session_state.drum_slots = {}

# --- FUNCTII AUDIO ---
def load_and_fix(path, sr=44100):
    # Forțăm Mono pentru a evita erorile de tip "Shape mismatch" la mixaj
    y, _ = librosa.load(path, sr=sr, mono=True)
    return librosa.util.normalize(y)

# --- SIDEBAR: DRUM LIBRARY (REDRUM SLOTS) ---
with st.sidebar:
    st.header("🥁 Drum Slots (Max 5)")
    st.write("Încarcă sample-uri de tobe (8-12 secunde).")
    up_drums = st.file_uploader("Sloturi active:", type=['mp3', 'wav'], accept_multiple_files=True)
    
    if up_drums:
        for d in up_drums[:5]:
            if d.name not in st.session_state.drum_slots:
                t = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                t.write(d.getbuffer())
                st.session_state.drum_slots[d.name] = t.name
        st.success(f"Tobe gata de utilizare: {len(st.session_state.drum_slots)}")

# --- ZONA DE UPLOAD MELODII ---
st.markdown("### 1. Încarcă Melodiile")
files = st.file_uploader("Selectează piesele pentru mixaj:", type=['mp3', 'wav'], accept_multiple_files=True)

if files:
    for f in files:
        if not any(t['nume'] == f.name for t in st.session_state.tracks):
            t = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            t.write(f.getbuffer())
            
            try:
                # Analiză BPM sigură (Fix pentru eroarea TypeError)
                y_p, _ = librosa.load(t.name, sr=22050, duration=30)
                tempo, _ = librosa.beat.beat_track(y=y_p, sr=22050)
                # Extragem valoarea indiferent de formatul returnat de librosa
                bpm_val = float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo)
            except:
                bpm_val = 120.0 # Valoare de rezervă
            
            st.session_state.tracks.append({
                "nume": f.name, 
                "path": t.name, 
                "bpm": round(bpm_val, 1),
                "drum_loop": "Fără (Crossfade)", 
                "durata": 60, 
                "start": 0
            })
    st.rerun()

# --- INTERFATA DE CONFIGURARE MIX ---
if st.session_state.tracks:
    st.markdown("### 2. Configurează Ordinea și Tranzițiile")
    
    for i, track in enumerate(st.session_state.tracks):
        with st.container(border=True):
            col_info, col_drums, col_settings, col_del = st.columns([3, 2, 2, 0.5])
            
            with col_info:
                st.markdown(f"**{i+1}. {track['nume']}**")
                st.caption(f"BPM Detectat: {track['bpm']}")
            
            with col_drums:
                drum_opts = ["Fără (Crossfade)"] + list(st.session_state.drum_slots.keys())
                st.session_state.tracks[i]['drum_loop'] = st.selectbox(
                    f"Tobe spre piesa {i+2 if i < len(st.session_state.tracks)-1 else 'Final'}:", 
                    options=drum_opts, 
                    key=f"d_{i}"
                )
            
            with col_settings:
                st.session_state.tracks[i]['durata'] = st.number_input("Durată (sec)", 10, 600, int(track['durata']), key=f"t_{i}")
            
            with col_del:
                if st.button("🗑️", key=f"rm_{i}"):
                    st.session_state.tracks.pop(i)
                    st.rerun()

    # --- BUTON GENERARE MIX FINAL ---
    st.markdown("---")
    if st.button("🚀 GENEREAZĂ MIXUL FINAL CU REDRUM", type="primary"):
        try:
            with st.spinner("Mixare profesională în curs... Se procesează straturile audio."):
                sr = 44100
                final_audio = np.array([], dtype=np.float32)
                
                for i, row in enumerate(st.session_state.tracks):
                    # Încărcare segment piesă
                    y = load_and_fix(row['path'], sr=sr)
                    start_s = int(row['start'] * sr)
                    end_s = start_s + int(row['durata'] * sr)
                    y_cut = y[start_s:end_s]
                    
                    # Aplicare Redrum (Tobe peste tranziție)
                    if row['drum_loop'] != "Fără (Crossfade)":
                        d_path = st.session_state.drum_slots[row['drum_loop']]
                        y_drum = load_and_fix(d_path, sr=sr)
                        
                        # Lungime suprapunere (max 8 secunde)
                        overlap = min(len(y_drum), int(8 * sr), len(y_cut))
                        # Mixaj: scadem volumul melodiei pentru a lăsa tobele să iasă în evidență
                        y_cut[-overlap:] = (y_cut[-overlap:] * 0.4) + (y_drum[:overlap] * 0.7)
                    
                    # Concatenare cu mic crossfade (1.5 secunde) pentru fluiditate
                    if i == 0:
                        final_audio = y_cut
                    else:
                        cf_len = int(1.5 * sr)
                        if len(final_audio) > cf_len:
                            fade_out = np.linspace(1, 0, cf_len)
                            fade_in = np.linspace(0, 1, cf_len)
                            mixed_zone = (final_audio[-cf_len:] * fade_out) + (y_cut[:cf_len] * fade_in)
                            final_audio = np.concatenate([final_audio[:-cf_len], mixed_zone, y_cut[cf_len:]])
                        else:
                            final_audio = np.concatenate([final_audio, y_cut])

                # Salvare și Export
                out_name = "SmartMix_Redrum_Result.wav"
                sf.write(out_name, final_audio, sr)
                
                st.success("✅ Mixaj finalizat cu succes!")
                st.audio(out_name)
                with open(out_name, "rb") as f_res:
                    st.download_button("💾 DESCARCĂ MIXUL COMPLET", f_res, file_name=out_name)

        except Exception as e:
            st.error(f"Eroare tehnică la mixaj: {e}")
            st.info("Sfat: Verifică dacă toate fișierele audio sunt valide și au o durată minimă de 10 secunde.")

else:
    st.info("Începe prin a încărca câteva piese și cel puțin un loop de tobe în bara laterală.")
