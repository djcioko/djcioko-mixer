# adauga langa importuri daca vrei Camelot
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Camelot: major = B, minor = A
CAMELOT_MAJOR = {0: '8B', 1: '3B', 2: '10B', 3: '5B', 4: '12B', 5: '7B',
                 6: '2B', 7: '9B', 8: '4B', 9: '11B', 10: '6B', 11: '1B'}
CAMELOT_MINOR = {0: '5A', 1: '12A', 2: '7A', 3: '2A', 4: '9A', 5: '4A',
                 6: '11A', 7: '6A', 8: '1A', 9: '8A', 10: '3A', 11: '10A'}

def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    # profil major / minor (Krumhansl-ish, simplificat)
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    best, best_score = "C", -1e9
    best_idx, best_mode = 0, "major"
    for i in range(12):
        s_maj = float(np.corrcoef(np.roll(chroma_mean, -i), major)[0, 1])
        s_min = float(np.corrcoef(np.roll(chroma_mean, -i), minor)[0, 1])
        if s_maj > best_score:
            best_score, best_idx, best_mode = s_maj, i, "major"
            best = NOTES[i]
        if s_min > best_score:
            best_score, best_idx, best_mode = s_min, i, "minor"
            best = NOTES[i] + "m"
    camelot = CAMELOT_MAJOR[best_idx] if best_mode == "major" else CAMELOT_MINOR[best_idx]
    return best, camelot

def analyze_track(path):
    y, sr = librosa.load(path, sr=22050, duration=60)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)[0]
    thr = np.mean(rmse) * 1.3
    hits = np.where(rmse > thr)[0]
    start_s = librosa.frames_to_time(hits[0], sr=sr) if len(hits) else 0.0
    key, camelot = detect_key(y, sr)
    return round(float(tempo), 1), round(float(start_s), 2), key, camelot
