from pathlib import Path
import numpy as np

BASE_DATA = Path("data")
BEHAVIOURAL_DIR = BASE_DATA / "behavioural_data"
STIMULI_DIR = BASE_DATA / "stimuli"
EEG_DIR = BASE_DATA / "EEG/raw_eeg"
PREPROCESSED_DIR = EEG_DIR.parent / "preprocessed"
PREPROCESSED_LISTENING_DIR = BASE_DATA / "preprocessed_listening"
ONSETS_DIR = STIMULI_DIR / "processed_audios/onsets"
LOG_DIR = Path("detailed_logs")
LOG_TO_FILE = True
LOG_LEVEL ="INFO" # "INFO" "DEBUG" "WARNING" "ERROR" "CRITICAL"

ATTRIBUTE_PREPROCESS = 'Standarize'
EEG_PREPROCESS = 'Standarize'
SOLVER = 'ridge'  # 'ridge' or 'boosting'
N_FOLDS = 5

VALIDATION_LIMIT_PERCENTAGE = 0.02
DEFAULT_ALPHA = 1.0
SET_ALPHA = False
min_order, max_order, steps, base_log = -4, 5, 48, 10
ALPHAS_GRID = np.logspace(
    min_order, 
    max_order, 
    steps, 
    base=base_log
)
alpha_step = np.diff(np.log(ALPHAS_GRID))[0]


# Time lags and delays
NUMBER_OF_CHANNELS = 64
TARGET_SAMPLING_RATE = 128  # Hz
TMIN, TMAX = -.2, .6
DELAYS = np.arange(int(np.round(TMIN * TARGET_SAMPLING_RATE)), int(np.round(TMAX * TARGET_SAMPLING_RATE) + 1))
TIMES = (DELAYS/TARGET_SAMPLING_RATE)

FIGURES_DIR = Path("figures")
OVERWRITE_FIGURES = True
USE_SCIENCE_PLOTS = False
OUTPUT_DIR = Path("output")
OVERWRITE_RESULTS = True
VALIDATION_DIR = OUTPUT_DIR / "validation" / f'tmin{TMIN}_tmax{TMAX}' 
CORRELATIONS_DIR = OUTPUT_DIR / "correlations" / f'tmin{TMIN}_tmax{TMAX}'
TRFS_DIR = OUTPUT_DIR / "trfs" / f'tmin{TMIN}_tmax{TMAX}'
CHANNEL_SELECTION = None # to get all channels, else give explicit list (Fcz, C3, etc)
OVERWRITE_EXISTING_ATTRIBUTES = False

SIDES = [
    'mono',
    # 'left',
    # 'right'
    ]  

BAND_FREQ = [
    'Broad',
    'Delta',
    'Theta',
    'Alpha',
    'Beta',
    'FullBand'
    ]  
ATTRIBUTES = [
    'Envelope', 
    'Spectrogram', 
    'BipOnsets'
    # 'Phonemes'
]
ATTRIBUTE_PARAMS = {
    'Envelope': {
    },
    'Spectrogram': {
        'target_sample_rate': TARGET_SAMPLING_RATE,
        'n_mels': 21
    },
    'Phonemes': {
    }
}
SUBJECTS = [sub_path.stem.replace("_preprocessed", "") for sub_path in list((PREPROCESSED_DIR / 'fif').glob("*_preprocessed.fif"))]


VERBOSE_LEVEL = 'CRITICAL'
RANDOM_SEED = 42
