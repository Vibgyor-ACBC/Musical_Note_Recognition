import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
from collections import defaultdict, Counter
import matplotlib.animation as animation
from pathlib import Path
import json
import os
import bisect


# ---------- STEP 1: STFT ----------
def stft(signal, sr, fft_size=2048, hop_size=512, window=np.hanning):
    win = window(fft_size)
    num_frames = 1 + (len(signal) - fft_size) // hop_size
    stft_matrix = np.zeros((fft_size // 2 + 1, num_frames), dtype=np.complex64)

    for i in range(num_frames):
        start = i * hop_size
        frame = signal[start:start + fft_size] * win
        spectrum = np.fft.rfft(frame)
        stft_matrix[:, i] = spectrum

    magnitude = np.abs(stft_matrix)
    freq_bins = np.fft.rfftfreq(fft_size, 1.0 / sr)
    time_bins = np.arange(num_frames) * hop_size / sr
    return magnitude, freq_bins, time_bins


# ---------- STEP 2: CONSTELLATION MAP ----------
def get_constellation_map(magnitude, freq_bins, time_bins,
                          prominence_db=30, max_peaks=5, max_freq=4000):
    """
    Build a constellation list of peak tuples: (time, freq, prominence_db)

    Parameters
    ----------
    magnitude : 2D array
        Linear magnitude matrix from stft (shape [freq_bins, time_bins])
    freq_bins, time_bins : arrays
        Frequency and time arrays returned by stft
    prominence_db : float
        Threshold for peak prominence in dB
    max_peaks : int
        Pick up to this many peaks per frame (by prominence)
    max_freq : float
        Ignore peaks above this frequency (Hz)

    Returns
    -------
    constellation : list of tuples
        (time, freq, prom_db)
    """
    constellation = []
    # convert to dB for peak picking
    mag_db = 20 * np.log10(magnitude + 1e-10)

    for t_idx, frame_db in enumerate(mag_db.T):
        # find all peaks on dB-scaled frame
        peaks, props = find_peaks(frame_db, prominence=prominence_db)
        if len(peaks) == 0:
            continue

        prominences = props.get("prominences", np.zeros_like(peaks))

        # filter peaks by max_freq BEFORE selecting top ones
        valid_mask = freq_bins[peaks] <= max_freq
        peaks = peaks[valid_mask]
        prominences = prominences[valid_mask]

        if len(peaks) == 0:
            continue

        # sort remaining peaks by prominence desc and pick top ones
        sorted_idx = np.argsort(prominences)[::-1]
        top_idx = sorted_idx[:max_peaks]

        for si in top_idx:
            p = peaks[si]
            freq = freq_bins[p]
            prom_db = prominences[si]
            time = float(time_bins[t_idx])
            constellation.append((time, float(freq), float(prom_db)))

    return constellation

# def time_frequency_filter(magnitude, smoothing=(5, 5), threshold_db=6):
#     """
#     Apply time-frequency filtering:
#     1. Estimate noise floor (median over time per frequency bin).
#     2. Build a mask where signal > noise + threshold.
#     3. Smooth mask with median filter.
#     """
#     mag_db = 20 * np.log10(magnitude + 1e-10)

#     # Estimate noise floor (per frequency bin)
#     noise_floor = np.median(mag_db, axis=1, keepdims=True)

#     # Mask bins significantly above noise
#     mask = mag_db > (noise_floor + threshold_db)

#     # Median filter the mask to remove isolated speckles
#     mask = median_filter(mask.astype(float), size=smoothing)

#     # Apply mask back to magnitude
#     attenuation = 0
#     filtered_magnitude = magnitude * (mask + attenuation*(1-mask))
#     return filtered_magnitude
def time_frequency_filter(magnitude, filter_size=(3, 3)):
    """
    Applies a median filter to the magnitude spectrogram to reduce noise.

    Parameters
    ----------
    magnitude : 2D array
        Linear magnitude matrix from stft (shape [freq_bins, time_bins])
    filter_size : tuple (f_len, t_len)
        Size of the median filter in (frequency, time) bins.

    Returns
    -------
    filtered_magnitude : 2D array
        Noise-reduced magnitude spectrogram.
    """
    # Use median_filter for non-linear noise reduction
    filtered_magnitude = median_filter(magnitude, size=filter_size)
    return filtered_magnitude



# ---------- STEP 3: HASHING ----------
def generate_hashes(constellation, fan_out=4, max_freq=5000):
    """
    Create hashes from constellation map:
    (f1, f2, Δt) → anchor time

    Parameters
    ----------
    constellation : list of tuples
        Each tuple is (time, freq, prom_db)
    fan_out : int
        Number of target points to pair with each anchor point
    max_freq : float
        Ignore peaks/hashes above this frequency (Hz)

    Returns
    -------
    hashes : list of tuples
        [( (f1, f2, Δt), t1 ), ...]
    """
    def bin_frequency(f):
        # Calculate the binned value, then explicitly cast to Python's native int
        return int(int(f) // bin_size * bin_size)
    def bin_timediff(t):
        return float(float(t)//time_size * time_size)
    hashes = []
    bin_size = 50
    time_size = 0.02
    for i in range(len(constellation)):
        t1, f1, _ = constellation[i]
        if f1 > max_freq:
            continue
        f1_binned = bin_frequency(f1)
        for j in range(1, fan_out + 1):
            if i + j < len(constellation):
                t2, f2, _ = constellation[i + j]
                if f2 > max_freq:
                    continue
                f2_binned = bin_frequency(f2)
                dt = t2 - t1
                if 0 < dt < 3.0:  # limit Δt window
                    hash_val = (f1_binned, f2_binned, round(dt, 2))
                    hashes.append((hash_val, round(t1, 2)))
    return hashes

HASH_DB_PATH = "hashes_db.json"

def load_hash_db(path=HASH_DB_PATH):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        # corrupt file or read error -> treat as empty
        return {}
    # convert stored lists back to tuples where appropriate
    saved = {}
    for song_id, entries in raw.items():
        converted = []
        for item in entries:
            try:
                h_list, t = item
                h_tuple = tuple(h_list)
                converted.append((h_tuple, float(t)))
            except Exception:
                continue
        if converted:
            saved[song_id] = converted
    return saved

def save_hash_db(saved, path=HASH_DB_PATH):
    # convert tuples to lists for JSON
    raw = {}
    for song_id, entries in saved.items():
        raw_entries = []
        for (h, t) in entries:
            raw_entries.append([list(h), float(t)])
        raw[song_id] = raw_entries
    # write atomically
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)
    os.replace(tmp, path)

# ---------- STEP 4: DATABASE ----------
class FingerprintDB:
    def __init__(self,dt_tolerance=0.03, f_tolerance = 25):
        self.db = defaultdict(list)  # (f1,f2,dt) → [(song_id, t_anchor)]
        self.dt_tol = dt_tolerance
        self.f_tol = f_tolerance

    def add_song(self, song_id, hashes):
        for h, t in hashes:
            self.db[h].append((song_id, t))

    def match(self, query_hashes):
        matches = []
        for h, qt in query_hashes:
            # print(self.db)
            if h in self.db:
                for song_id, st in self.db[h]:
                    offset = round(st - qt, 2)  # time alignment
                    matches.append((song_id, offset))
        return matches
        # dt_tol = 0.01
        # f_tol = self.f_tol

        # for (f1q, f2q, dtq), tq in query_hashes:
        #     # collect all keys that are near (f1q, f2q, dtq)
        #     for (f1r, f2r, dtr), entries in self.db.items():
        #         if abs(f1q - f1r) > f_tol or abs(f2q - f2r) > f_tol:
        #             continue

        #         # Time difference tolerance
        #         if abs(dtr - dtq) <= dt_tol:
        #             for song_id, st in entries:
        #                 offset = round(st - tq, 2)
        #                 matches.append((song_id, offset))

        # for (f1q, f2q, dtq), tq in query_hashes:

        # # search nearby keys instead of full scan
        #     for df1 in range(-f_tol, f_tol + 1):
        #         f1_candidate = f1q + df1

        #         for df2 in range(-f_tol, f_tol + 1):
        #             f2_candidate = f2q + df2

        #         # loop small real-valued bins for dt
        #         # for example ±50 steps of 0.01 around dtq
        #             for k in range(-1, 2):
        #                 dt_candidate = round(dtq + k * dt_tol, 2)

        #                 key = (f1_candidate, f2_candidate, dt_candidate)

        #                 if key not in self.db:
        #                     continue

        #                 for song_id, st in self.db[key]:
        #                     offset = round(st - tq, 2)
        #                     matches.append((song_id, offset))
        # return matches



# ---------- STEP 5: SONG IDENTIFICATION ----------
def identify_song(db, query_hashes):
    matches = db.match(query_hashes)
    if not matches:
        return None

    counter = Counter(matches)
    best_match, votes = counter.most_common(1)[0]
    print(counter.most_common(20))
    song_id, offset = best_match
    return song_id, votes

def compute_frequency_error(orig_const, query_const, time_tolerance=0.05):
    orig_times = np.array([t for t, f, p in orig_const])
    orig_freqs = np.array([f for t, f, p in orig_const])

    times = []
    errors = []

    for tq, fq, _ in query_const:
        idx = np.searchsorted(orig_times, tq)

        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(orig_times):
            candidates.append(idx)

        best = None
        best_dt = float("inf")

        for c in candidates:
            dt = abs(orig_times[c] - tq)
            if dt < best_dt and dt <= time_tolerance:
                best_dt = dt
                best = c

        if best is not None:
            freq_error = fq - orig_freqs[best]
            times.append(tq)
            errors.append(freq_error)

    return np.array(times), np.array(errors)


def dominant_frequency_curve(magnitude, freq_bins):
    """
    Returns an array of dominant frequencies over time.
    dominant frequency = frequency with maximum magnitude in each frame.
    """
    dom_freqs = []
    for frame in magnitude.T:
        idx = np.argmax(frame)
        dom_freqs.append(freq_bins[idx])
    return np.array(dom_freqs)

def plot_curve(orig_time_bins,orig_curve,query_time_bins,query_curve):
    plt.figure(figsize=(14, 6))

    plt.plot(orig_time_bins, orig_curve, label="Original Audio", linewidth=1.5)
    plt.plot(query_time_bins, query_curve, label="Query Audio", linewidth=1.5)

    plt.xlabel("Time (s)")
    plt.ylabel("Dominant Frequency (Hz)")
    plt.title("Frequency Variation: Original vs Query Audio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def animate_constellation(signal, sr, constellation, hashes, save_path=None, interval=200):
    """
    Animate constellation map + hashes.
    - signal: waveform
    - sr: sample rate
    - constellation: [(time, freq), ...]
    - hashes: [(hash_val, t_anchor), ...]
    - save_path: if given, save animation as mp4/gif
    - interval: ms between frames
    """

    # Spectrogram background
    S = librosa.stft(signal, n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    fig, ax = plt.subplots(figsize=(12, 6))
    librosa.display.specshow(S_db, sr=sr, hop_length=512,
                             x_axis='time', y_axis='log',
                             cmap='magma', ax=ax)

    ax.set_title("Animated Constellation Map + Hashes")
    ax.set_ylim(100, sr // 2)

    # Initialize plot elements
    peak_scatter = ax.scatter([], [], c='cyan', marker='.', s=20, label="Peaks")
    arrows = []
    ax.legend(loc="upper right")

    # Data prep
    times = [t for (t, f) in constellation]
    freqs = [f for (t, f) in constellation]

    def init():
        peak_scatter.set_offsets(np.empty((0,2)))
        return [peak_scatter]

    def update(frame_idx):
        # Show peaks up to current frame
        peak_scatter.set_offsets(np.column_stack((times[:frame_idx+1],
                                                  freqs[:frame_idx+1])))

        # Add arrows for hashes whose anchor time <= this frame
        while len(arrows) < len(hashes) and hashes[len(arrows)][1] <= times[frame_idx]:
            h, t1 = hashes[len(arrows)]
            f1, f2, dt = h
            t2 = t1 + dt
            arr = ax.arrow(t1, f1, (t2 - t1), (f2 - f1),
                           head_width=50, head_length=0.1,
                           color="yellow", alpha=0.6, length_includes_head=True)
            arrows.append(arr)

        return [peak_scatter] + arrows

    ani = animation.FuncAnimation(fig, update, frames=len(constellation),
                                  init_func=init, blit=False, interval=interval,
                                  repeat=False)

    if save_path:
        ani.save(save_path, fps=5, dpi=150)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

    return ani

def recognize_uploaded_song(file_path="/home/vibgyor/BTP/Musical_Note_Recognition/query/jeena_jeena_recording.mp3"):
    """
    file_path: path to uploaded audio file
    Returns: best match song_id or None
    """
    MUSIC_DIR = "/home/vibgyor/BTP/Musical_Note_Recognition/music_demo"
    db = FingerprintDB()
    saved_hashes = load_hash_db()

    # Dynamically build songs dict from all .mp3 files in the directory
    songs = {}
    for filename in os.listdir(MUSIC_DIR):
        if filename.lower().endswith(".mp3"):
            song_name = Path(filename).stem.replace("_", " ")   # e.g. Jeena_Jeena → Jeena Jeena
            songs[song_name] = os.path.join(MUSIC_DIR, filename)

    # Step 1: Build fingerprint DB
    for song_id, path in songs.items():
        if song_id in saved_hashes:
            db.add_song(song_id, saved_hashes[song_id])
            print(f"Loaded saved hashes for {song_id} ({len(saved_hashes[song_id])} hashes)")
            continue

        try:
            sig, sr = librosa.load(path, sr=None, mono=True)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue

        # Normalize
        peak = np.max(np.abs(sig)) if sig.size else 0.0
        if peak > 0:
            sig = sig / peak
        else:
            print(f"Warning: {path} appears silent, skipping.")
            continue

        # Generate hashes
        mag, freqs, times = stft(sig, sr)
        filtered_mag = time_frequency_filter(mag, filter_size=(3, 3)) 
        const_map = get_constellation_map(filtered_mag, freqs, times)
        hashes = generate_hashes(const_map)

        db.add_song(song_id, hashes)
        print(f"Indexed {song_id} with {len(hashes)} hashes")

        # Save hashes for persistence
        saved_hashes[song_id] = [(tuple(h), float(t)) for h, t in hashes]
        save_hash_db(saved_hashes)

    # Step 2: Process uploaded file
    # if file_path is not None:
    song_path = Path(file_path)
    query_sig, sr = librosa.load(song_path, sr=None, mono=True)
    # else:
    #     print("🎤 Listening... play music now!")
    #     recording = sd.rec(int(duration * sr), samplerate=sr,
    #                        channels=1, dtype='float32')
    #     sd.wait()
    #     query_sig = recording.flatten()
    #     print("✔ Audio captured, processing...")
    query_sig /= np.max(np.abs(query_sig))
    mag, freqs, times = stft(query_sig, sr)
    filtered_mag = time_frequency_filter(mag, filter_size=(3, 3)) 
    const_map = get_constellation_map(filtered_mag, freqs, times)
    query_hashes = generate_hashes(const_map)
    print(f"Indexed {song_path.stem} with {len(query_hashes)} hashes")

    # Step 3: Identify
    result = identify_song(db, query_hashes)
    if result:
        print(f"Best match {result[0]} with {result[1]} votes")
        return result[0]
    else:
        return "No match found"

# ---------- DEMO ----------
if __name__ == "__main__":
    print(recognize_uploaded_song())

    # Database
    # db = FingerprintDB()

    # # Songs to index (replace with your mp3s)
    # songs = {
    #     "song1": "/home/vibgyor/BTP/musical/music/017_TujheDekhaToh.mp3",
    #     # "song2": "/home/vibgyor/BTP/musical/music/010_DheereDheere.mp3",
    #     # "song3": "/home/vibgyor/BTP/musical/music/001_6AM.mp3",
    #     # "song4": "/home/vibgyor/BTP/musical/music/003_AgarTumSaathHo.mp3",
    #     # "song5": "/home/vibgyor/BTP/musical/music/009_DesiKalakaar.mp3",
    #     # "song6": "/home/vibgyor/BTP/musical/music/011_HoGyaHaiTujhko.mp3",
    #     # "song7": "/home/vibgyor/BTP/musical/music/014_Pachtaoge.mp3",
    #     # "song8": "/home/vibgyor/BTP/musical/music/004_Alagaasman.mp3",
    #     "song9": "/home/vibgyor/BTP/musical/music/012_JeenaJeena.mp3",
    #     "song10": "/home/vibgyor/BTP/musical/music/008_ChaarKadam.mp3",
    #     "song11": "/home/vibgyor/BTP/musical/music/007_ChaandBaaliyan.mp3",
    #     # "song12": "/home/vibgyor/BTP/musical/music/002_AbhinaJaoChhodKar.mp3",
    #     # "song13": "/home/vibgyor/BTP/musical/music/005_BadeAchheLagteHain.mp3",
    #     # "song14": "/home/vibgyor/BTP/musical/music/015_PalPalDilKePaas.mp3",
    #     # "song15": "/home/vibgyor/BTP/musical/music/020_YeSamma.mp3",
    #     # "song16": "/home/vibgyor/BTP/musical/music/021_YehRateinYehMausam.mp3",
    # }

    # # Index songs
    # saved_hases = load_hash_db()
    # song_const={}
    # for song_id, path in songs.items():
    #     sig, sr = librosa.load(path, sr=None, mono=True)
    #     sig /= np.max(np.abs(sig))

    #     mag, freqs, times = stft(sig, sr)
    #     # filtered_magnitude = time_frequency_filter(mag,smoothing=(7, 7),threshold_db=6)
    #     song_const[song_id]=[mag,freqs,times]
    
    #     filtered_mag = time_frequency_filter(mag, filter_size=(3, 3)) 
        
    #     const_map = get_constellation_map(filtered_mag, freqs, times)
    #     hashes = generate_hashes(const_map)
    #     db.add_song(song_id, hashes)
    #     print(f"Indexed {song_id} with {len(hashes)} hashes")

    # # Query (snippet of song1)
    # song_path = Path("/home/vibgyor/BTP/musical/query/jeena_jeena_recording.mp3")
    # query_sig, sr = librosa.load(song_path, sr=None, mono=True)
    # query_sig /= np.max(np.abs(query_sig))

    # mag, freqs, times = stft(query_sig, sr)
    # # filtered_magnitude = time_frequency_filter(mag,smoothing=(7, 7),threshold_db=6)
    
    # filtered_mag = time_frequency_filter(mag, filter_size=(3, 3)) 
    
    # const_map = get_constellation_map(filtered_mag, freqs, times)
    # query_hashes = generate_hashes(const_map)
    # print(f"Indexed {song_path.stem} with {len(query_hashes)} hashes")

    # result = identify_song(db, query_hashes)
    # if result:
    #     print(f"Best match: {result[0]} with {result[1]} votes")
    #     # time_err,freq_err=compute_frequency_error(const_map,song_const[result[0]])
    #     original_curve= dominant_frequency_curve(song_const[result[0]][0],song_const[result[0]][1])
    #     query_curve= dominant_frequency_curve(mag,freqs)
    #     # plot_curve(original_curve,song_const[result[0]][2],query_curve,times)

    # else:
    #     print("No match found")

    # # Animate (show live)
    # animate_constellation(query_sig, sr, const_map, hashes, save_path=None)

    # Or save to mp4/gif
    # animate_constellation(y, sr, const_map, hashes, save_path="constellation.mp4")

