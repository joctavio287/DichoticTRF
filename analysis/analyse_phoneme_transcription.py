"""
Analyze and visualize phoneme-level transcriptions alongside phoneme-level PLLRs.
Generates plots for each audio file segment, showing PLLR heatmaps with overlaid
DNN-generated transcriptions.
# TODO : it can be misalignments between DNN and PLLR, need to fix that
"""

import warnings
from pathlib import Path
import numpy as np
import sys
sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)
from utils.helpers_audio import (
    LEXICON, transcribe_audio,
    save_wav, _get_dnn_instance,
    read_ogg
)
from utils.helpers_processing import normalize_text
import config

import matplotlib.pyplot as plt
import librosa

warnings.filterwarnings("ignore")
PLOT_ONLY_ONE_COMPONENT = True
SEGMENT_DURATION = 3.5  # seconds
DNN_OFFSET = 0 #-0.1  # seconds to compensate for DNN transcription delay
PHONEME_LABELS = LEXICON['phonemes']
DNN_MODEL = 'base'  # tiny, base, small, medium, large
DNN_CACHE = {}
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

for side in ['left', 'right']:
    pllr_paths = list((config.PREPROCESSED_LISTENING_DIR / "phonemes" / side).glob("*.npz"))
    for idx, pllr_path in enumerate(pllr_paths):
        print(f"Processing side {side} {(idx + 1)*100/len(pllr_paths):.2f}%", end='\r')
        # Load data
        npz_data = np.load(pllr_path, allow_pickle=True)
        
        # Get audio filepath 
        if config.ATTRIBUTE_PARAMS['Phonemes']['use_unprobed_audio']:
            audio_filepath = npz_data["audio_filepath"].item().replace('with_probe', 'no_probe').replace('tone', 'no')
        else:
            audio_filepath = npz_data["audio_filepath"].item() 
        
        # Separate audio if needed
        sample_rate, audio_stereo = read_ogg(audio_filepath, return_sample_rate=True)
        if side == 'left':
            audio_filepath = TEMP_DIR / f"{pllr_path.stem}_left.wav"
            save_wav(audio_filepath, sample_rate=sample_rate, data=audio_stereo[:, 0])
        elif side == 'right':
            audio_filepath = TEMP_DIR / f"{pllr_path.stem}_right.wav"
            save_wav(audio_filepath, sample_rate=sample_rate, data=audio_stereo[:, 1])
        else:
            raise ValueError("For Fonogram, side must be 'left' or 'right'")

        # Get plain text transcription
        if idx % 10 == 0:
            DNN_CACHE = _get_dnn_instance({}, model=DNN_MODEL)
        else:
            DNN_CACHE = _get_dnn_instance(DNN_CACHE, model=DNN_MODEL)
        transcription = transcribe_audio(
            audio_filepath=audio_filepath,
            dnn_model=DNN_MODEL,
            dnn_cache=DNN_CACHE,
            timestamp=True
        )
        # Load pllr and sample rate
        pllr = npz_data["attribute_values"] # (n_timepoints, n_phonemes)
        greedy_phonemes = np.argmax(pllr, axis=1)
        sample_rate = npz_data["sample_rate"].item()
        total_duration = pllr.shape[0] / sample_rate
        hop_length = 1 # since pllr is already downsampled to match EEG

        # Divide into segments and plot
        n_segments = np.arange(
            int(np.ceil(total_duration / SEGMENT_DURATION))
        )
        np.random.shuffle(n_segments)
        for i in n_segments:
            start_time = i * SEGMENT_DURATION
            end_time = min((i + 1) * SEGMENT_DURATION, total_duration)
            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)

            fig, ax = plt.subplots(figsize=(12, 6))
            img = librosa.display.specshow(
                pllr[start_idx:end_idx].T, 
                x_axis='time', 
                sr=sample_rate, 
                hop_length=hop_length, 
                cmap='magma', 
                ax=ax
            )
            ax.set_yticks(np.arange(len(PHONEME_LABELS)))
            ax.set_yticklabels(PHONEME_LABELS)
            ax.set_ylabel('Phonemes')
            ax.set_xlabel('Time (s)')
            
            greedy_phonemes = np.argmax(pllr[start_idx:end_idx], axis=1)
            times = (np.arange(start_idx, end_idx) / sample_rate)
            ax.scatter(
                times - start_time,  # align to segment start
                greedy_phonemes,
                color='white',
                s=8,
                marker='o',
                label='Greedy Transcription'
            )

            for segment in transcription:
                for word_info in segment['words']:
                    if word_info['start'] >= start_time and word_info['end'] <= end_time:
                        center = (word_info['start'] + word_info['end']) / 2 - start_time + DNN_OFFSET
                        ax.text(
                            center, 
                            len(PHONEME_LABELS) + 0.5, 
                            normalize_text(word_info['word']), 
                            ha='center', 
                            va='bottom', 
                            bbox=dict(facecolor='white', alpha=0.5)
                        )
            ax.legend(loc='upper right')

            fig.colorbar(img, ax=ax, label='Log-Likelihood')
            fig.suptitle(f"{audio_filepath.stem} [{start_time:.1f}s - {end_time:.1f}s]")
            fig.tight_layout()
            save_path = config.FIGURES_DIR / "analysis" / "phoneme_transcriptions" /  f"phoneme_transcription_{pllr_path.stem}_segment_{i+1}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                save_path
            )
            plt.close(fig)
            if PLOT_ONLY_ONE_COMPONENT:
                break

# Clean up temporary files
for temp_file in TEMP_DIR.glob("*.wav"):
    temp_file.unlink()
TEMP_DIR.rmdir()