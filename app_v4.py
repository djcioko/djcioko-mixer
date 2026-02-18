import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import os
import tempfile

st.set_page_config(page_title="SmartMix V4 - Redrum Edition", layout="wide", page_icon="🥁")

# CSS pentru un look mai pro
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    .track-card { padding: 15px; border-radius: 10px; border: 1px solid #31333f; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎧 SmartMix Pro V4.0")
st.subheader("Manual Mix & Redrum Transition")

# --- INITIALIZARE SESSION STATE ---
if 'tracks' not in st.session_state:
    st.session_state.tracks = []
if 'drum_slots' not in st.session_state:
    st.session_state.drum_slots = {}

# --- FUNCTII DE PROCESARE ---
def load_and_fix(path, sr=44100):
    y, _ = librosa.load(path, sr=sr, mono=True) # Forțăm Mono pentru stabilitate la mixaj
    return y

# --- SIDEBAR: DRUM LIBRARY ---
with st.sidebar:
    st.header("🥁 Drum Slots (Max 5)")
    up_drums = st.file_uploader("Încarcă sample-uri de 10s:", type=['mp3', 'wav'], accept_multiple_files=True)
    
    if up_drums:
        for d in up_drums[:5]:
            if d.name not in st.session_state.drum_slots:
                t = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                t.write(d.getbuffer())
                st.session_state.drum_slots[d.name] = t.name
        st.success(f"Sloturi active: {len(st.session_state.drum_slots)}")

# --- ZONA DE UPLOAD ---
files = st.file_uploader("Încarcă melodiile principale:", type=['mp3', 'wav'], accept_multiple_files=True)

if files:
    for f in files:
        if not any(t['nume'] == f.name for t in st.session_state.tracks):
            t = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            t.write(f.getbuffer())
            
            # Analiză BPM rapidă
            y_p, _ = librosa.load(t.name, sr=22050, duration=20)
            bpm, _ = librosa.beat.beat_track(y=y_p, sr=22050)
            
            st.session_state.tracks.append({
                "nume": f.name, "path": t.name, "bpm": round(float(bpm), 1),
                "drum_loop": "Fără (Crossfade)", "durata": 60, "start": 0
            })
    st.rerun()

# --- INTERFATA DE ARANJARE ---
if st.session_state.tracks:
    st.write("---")
    for i, track in enumerate(st.session_state.tracks):
        with st.container():
            st.markdown(f"**Piesa {i+1}: {track['nume']}**")
            c1, c2, c3 = st.columns([2, 2, 1])
            
            with c1:
                drum_opts = ["Fără (Crossfade)"] + list(st.session_state.drum_slots.keys())
                st.session_state.tracks[i]['drum_loop'] = st.selectbox(f"Tobe la final:", drum_opts, key=f"d_{i}")
            
            with c2:
                st.session_state.tracks[i]['durata'] = st.number_input("Durată (sec)", 10, 600, int(track['durata']), key=f"t_{i}")
            
            with c3:
                if st.button("🗑️", key=f"rm_{i}"):
                    st.session_state.tracks.pop(i)
                    st.rerun()

    # --- BUTON GENERARE ---
    if st.button("🚀 GENEREAZĂ MIXUL FINAL"):
        try:
            with st.spinner("Se procesează mixajul audio..."):
                sr = 44100
                final_audio = np.array([], dtype=np.float32)
                
                for i, row in enumerate(st.session_state.tracks):
                    # Încărcăm piesa
                    y = load_and_fix(row['path'], sr=sr)
                    start_s = int(row['start'] * sr)
                    end_s = start_s + int(row['durata'] * sr)
                    y_cut = y[start_s:end_s]
                    y_cut = librosa.util.normalize(y_cut) * 0.9
                    
                    # Logică Tranziție (Redrum)
                    if row['drum_loop'] != "Fără (Crossfade)":
                        d_path = st.session_state.drum_slots[row['drum_loop']]
                        y_drum = load_and_fix(d_path, sr=sr)
                        
                        # Suprapunere de 8 secunde (sau cât e lung loop-ul)
                        overlap = min(len(y_drum), int(8 * sr), len(y_cut))
                        y_cut[-overlap:] = (y_cut[-overlap:] * 0.4) + (y_drum[:overlap] * 0.7)
                    
                    # Concatenare cu mic crossfade de siguranță (1 sec)
                    if i == 0:
                        final_audio = y_cut
                    else:
                        cf_len = int(1 * sr)
                        if len(final_audio) > cf_len:
                            fade_out = np.linspace(1, 0, cf_len)
                            fade_in = np.linspace(0, 1, cf_len)
                            mixed_zone = (final_audio[-cf_len:] * fade_out) + (y_cut[:cf_len] * fade_in)
                            final_audio = np.concatenate([final_audio[:-cf_len], mixed_zone, y_cut[cf_len:]])
                        else:
                            final_audio = np.concatenate([final_audio, y_cut])

                # Export final
                sf.write("mix_final.wav", final_audio, sr)
                st.audio("mix_final.wav")
                with open("mix_final.wav", "rb") as f:
                    st.download_button("💾 DESCARCĂ REZULTATUL", f, "SmartMix_V4.wav")

        except Exception as e:
            st.error(f"Eroare tehnică: {e}. Încearcă să urci piese mai scurte.")
