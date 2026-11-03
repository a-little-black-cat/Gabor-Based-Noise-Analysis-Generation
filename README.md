# OffRoad Sim Gabor-Based Noise Analysis and Generation

*To install all requirements, run*
`pip install -r requirements.txt` from root directory

*To run:*
`python .\train_texture_gan.py`from root directory

Flags:
`python .\train_texture_gan.py -f`
where -f is

`--from-save-state` : enabling to continue training from saved state if it gets interrupted

or

`--gen-output-only` : to generate outputs from saved model

Running without flags starts a fresh training run and replaces the existing
`model_weights` directory, as in the original workflow. `--clear` is also
accepted as an explicit request for the same fresh-run behavior.

The generator upsamples with nearest-neighbor resize followed by 3x3
convolutions and PixelNorm, avoiding transposed-convolution checkerboards and
BatchNorm moving-statistic drift. Training samples random crops with horizontal
and vertical flips. Generated samples include a border margin that is cropped
before saving and scoring, and Optuna ranks trials by Gabor texture distance.

Training is memory-bounded by default: trials use patches up to 128px, three
Optuna trials, 16 batches per epoch, and batches of at most 32 images. You can
override these with `--max-patch-size`, `--trials`, `--steps-per-epoch`, and
`--epochs`. For example:

```powershell
python -u .\train_texture_gan.py --max-patch-size 64 --trials 3 --steps-per-epoch 16 --epochs 100
```

If a run stops, inspect `model_weights\training_state.json`. Resume with
`--from-save-state`; generate saved outputs without training with:

```powershell
python .\train_texture_gan.py --gen-output-only --num-images 4
```

## NVIDIA GPU training with WSL2

This project uses TensorFlow. Current TensorFlow releases do not use CUDA GPUs
from native Windows, so NVIDIA GPU training should run in WSL2 with Ubuntu.
The repository's requirements install the CUDA-enabled TensorFlow extra on
Linux and the regular Windows package on native Windows.

Install WSL2 from an elevated PowerShell prompt on Windows:

```powershell
wsl --install -d Ubuntu
```

After restarting and completing Ubuntu's first-time setup, run these commands
inside Ubuntu:

```bash
cd /mnt/c/Users/beren/OneDrive/Desktop/gabor-v-texton
sudo apt update
sudo apt install -y python3-venv python3-pip
mkdir -p ~/venvs ~/tmp
python3 -m venv ~/venvs/gabor-texton
source ~/venvs/gabor-texton/bin/activate
python -m pip install --upgrade pip
TMPDIR=~/tmp pip install --no-cache-dir -r requirements.txt
```

Keep the virtual environment in your WSL home directory rather than under
`/mnt/c`; this avoids filling the Windows-mounted temporary filesystem during
the large CUDA dependency install.

Verify that TensorFlow sees the GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Run training and require GPU execution:

```bash
python train_texture_gan.py --require-gpu
```

If the verification command prints an empty list, check `nvidia-smi` inside
Ubuntu and confirm that the Windows NVIDIA driver is current. Do not install a
separate Linux NVIDIA driver inside WSL2.
