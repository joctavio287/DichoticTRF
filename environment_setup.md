# Create the conda environemnt
conda create -n speech-encoding -c conda-forge python=3.12 ffmpeg -y
conda activate speech-encoding

# Verify ffmpeg integration
ffmpeg -version

# Install torch and torch audio. Adjust to your graphic card if necessary (https://download.pytorch.org/whl/)
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129

# Download phonet's fork compute PLLRs
mkdir libs
cd libs
git clone https://github.com/joctavio287/phonet
cd phonet 
pip install -e .

# Check that phonet doesn't interfere with torch (cuda)
python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}')"
python -c "import torch; import tensorflow as tf; import phonet; print('Everything seems OK.')"

# Install necessary packages
pip install pandas numpy matplotlib psutil requests scikit-learn ipython mne librosa pydub
python -c "import torch,tensorflow, phonet, pandas, numpy, matplotlib, psutil, requests, sklearn, IPython, mne, librosa, pydub; print('Everything seems OK.')"
