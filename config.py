from pathlib import Path
import numpy as np

from utils.helpers_processing import any_on_gpu

# === Time lags and delays ===
TARGET_SAMPLING_RATE = 128  # Hz
TMIN, TMAX = -.2, .6 # s
NUMBER_OF_CHANNELS = 64
DELAYS = np.arange(
    int(np.round(TMIN * TARGET_SAMPLING_RATE)), int(np.round(TMAX * TARGET_SAMPLING_RATE) + 1)
)
TIMES = (DELAYS / TARGET_SAMPLING_RATE)

# === Relevant directories ===
LOG_DIR = Path("detailed_logs")
FIGURES_DIR = Path("figures")
OUTPUT_DIR = Path("output")
BASE_DATA = Path("data")
PREPROCESSED_LISTENING_DIR = BASE_DATA / "preprocessed_listening"
BEHAVIOURAL_DIR = BASE_DATA / "behavioural_data"
STIMULI_DIR = BASE_DATA / "stimuli"
EEG_DIR = BASE_DATA / "EEG/raw_eeg"
PREPROCESSED_EEG_DIR = EEG_DIR.parent / "preprocessed"
ONSETS_DIR = STIMULI_DIR / "processed_audios/onsets"
CORRELATIONS_DIR = OUTPUT_DIR / "correlations" / f'tmin{TMIN}_tmax{TMAX}' 
VALIDATION_DIR = OUTPUT_DIR / "validation" / f'tmin{TMIN}_tmax{TMAX}' 
TRFS_DIR = OUTPUT_DIR / "trfs" / f'tmin{TMIN}_tmax{TMAX}' 

# === Logging ===
LOG_TO_FILE = True
LOG_LEVEL = "INFO"  # "INFO" "DEBUG" "WARNING" "ERROR" "CRITICAL"
VERBOSE_MNE_LEVEL = 'CRITICAL'

# === Model parameters ===
ATTRIBUTE_PREPROCESS = 'Standarize'
EEG_PREPROCESS = 'Standarize'
SOLVER = 'ridge' 
N_FOLDS = 5
VALIDATION_LIMIT_PERCENTAGE = 0.02
DEFAULT_ALPHA = 1.0
min_order, max_order, steps, base_log = -4, 5, 48, 10
ALPHAS_GRID = np.logspace(
    min_order, 
    max_order, 
    steps, 
    base=base_log
)
alpha_step = np.diff(np.log(ALPHAS_GRID))[0]

# === Output/plotting ===
PARALLEL_WORKERS_LOADING = -1
OVERWRITE_EXISTING_ATTRIBUTES = False
OVERWRITE_FIGURES = True
USE_SCIENCE_PLOTS = False
OVERWRITE_RESULTS = True
SET_ALPHA = False

# === Channel selection ===
# to get all channels, else give explicit list (Fcz, C3, etc)
CHANNEL_SELECTION = None  

# === Experimental parameters ===
ATTRIBUTES = [
    'Envelope',
    'Spectrogram',
    'BipOnsets'
    # 'Phonemes'
]
BAND_FREQ = [
    'Broad',
    # 'Delta',
    # 'Theta',
    # 'Alpha',
    # 'Beta',
    # 'FullBand'
]
SIDES = [
    'mono',
    # 'left',
    # 'right'
]
# Disable parallelism if using GPU to avoid conflicts
if any_on_gpu(ATTRIBUTES):
    PARALLEL_WORKERS_LOADING = 0

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

# === Subjects ===
SUBJECTS = [
    sub_path.stem.replace("_preprocessed", "") for sub_path in list((PREPROCESSED_EEG_DIR / 'fif').glob("*_preprocessed.fif"))
]

# === Miscellaneous ===
RANDOM_SEED = 42
