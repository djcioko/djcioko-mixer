import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import os
import subprocess

st.set_page_config(page_title="DJCIOKO MEGA-MIXER", layout="wide")
st.title("🎧 DJCIOKO - TikTok Video Mixer")

if 'tracks' not in st.session_state:
    st.session_state.tracks = []

# 1. UPLOAD
files = st.file_uploader("Urci cele 10 melodii aici:", type=['mp3', 'wav'], accept_multiple_files=True)
foto = st.file_uploader("🖼️ Urci poza pentru TikTok aici:", type=['jpg', 'png', 'jpeg'])

if files and st.button("🔍 ANALIZEAZĂ ȘI PREGĂTEȘTE MIX"):
    results = []
    for f in files:
        # Forțăm încărcarea întregii piese pentru a nu rata refrenul
        y, sr = librosa.load(f, sr=44100)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Găsim refrenul (căutăm cea mai zgomotoasă parte)
        energy = librosa.feature.rms(y=y)[0]
        duration_30s = int(30 * sr / 512)
        if len(energy) > duration_30s:
            max_idx = np.argmax([np.mean(energy[i:i+duration_30s]) for i in range(len(energy)-duration_30s)])
            start_sec = (max_idx * 512) / sr
        else:
            start_sec = 0
            
        results.append({
            "Nume": f.name,
            "BPM": round(float(tempo), 1),
            "Start": start_sec,
            "audio": y,
            "sr": sr
        })
    
    # SORTARE AUTOMATĂ BPM: MIC -> MARE
    st.session_state.tracks = sorted(results, key=lambda x: x['BPM'])
    st.success(f"✅ Am pregătit {len(st.session_state.tracks)} piese în ordine crescătoare!")

# 2. AFIȘARE TABEL
if st.session_state.tracks:
    df_display = pd.DataFrame(st.session_state.tracks)[["Nume", "BPM", "Start"]]
    st.table(df_display)

    # 3. BUTON MIX NOW (AUDIO + VIDEO)
    if st.button("🚀 MIX NOW & GENERATE VIDEO"):
        mix_final = []
        sr_mix = 44100
        
        for piesa in st.session_state.tracks:
            s_idx = int(piesa['Start'] * sr_mix)
            e_idx = s_idx + (30 * sr_mix) # Tăiem FIX 30 de secunde
            
            segment = piesa['audio'][s_idx:e_idx]
            
            # Dacă piesa e mai scurtă de 30s, o luăm pe toată
            if len(segment) < (30 * sr_mix):
                segment = piesa['audio'][:int(30 * sr_mix)]

            # Crossfade rapid (0.2s)
            fade = int(0.2 * sr_mix)
            if len(segment) > fade:
                segment[:fade] *= np.linspace(0, 1, fade)
                segment[-fade:] *= np.linspace(1, 0, fade)
            
            mix_final.extend(segment)
            
        # Salvare Audio Temporar
        audio_path = "temp_mix.mp3"
        sf.write(audio_path, np.array(mix_final), sr_mix)
        
        if foto:
            # Salvare poză temporară
            with open("temp_img.jpg", "wb") as img_file:
                img_file.write(foto.getbuffer())
            
            # COMANDA FFMEG PENTRU VIDEO (Poza + Audio)
            video_path = "DJCIOKO_VIDEO.mp4"
            cmd = f"ffmpeg -y -loop 1 -i temp_img.jpg -i {audio_path} -c:v libx264 -tune stillimage -c:a aac -b:a 192k -pix_fmt yuv420p -shortest {video_path}"
            subprocess.run(cmd, shell=True)
            
            with open(video_path, "rb") as v:
                st.download_button("⬇️ DESCARCĂ VIDEO TIKTOK", v, file_name="DJCIOKO_FINAL.mp4")
        else:
            with open(audio_path, "rb") as a:
                st.download_button("⬇️ DESCARCĂ DOAR AUDIO", a, file_name="DJCIOKO_MIX.mp3")
