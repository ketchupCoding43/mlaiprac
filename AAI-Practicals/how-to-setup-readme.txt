Ubuntu 24.10
sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev liblzma-dev curl git
curl https://pyenv.run | bash
export PYENV_ROOT="$HOME/.pyenv"; [[ -d "$PYENV_ROOT/bin" ]] && export PATH="$PYENV_ROOT/bin:$PATH"; eval "$(pyenv init - bash)"
exec $SHELL
pyenv --version
pyenv 2.6.20
sudo apt install -y tk-dev
Global install python 3.10
pyenv install 3.10.14 (Do not use sudo here)
pyenv versions
Output: 3.10.14
cd to project
pyenv local 3.10.14
python -m venv venv
Ubuntu
sudo add-apt-repository ppa:deadsnakes/ppa
Sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev python3.12 python3.12-venv python3.12-dev
python3.10 -m venv venv
source venv/bin/activate
python --version
pip --version
TMPDIR=$HOME pip install --no-cache-dir -r requirements.txt
python your_file.py

Windows
py -m venv venv (OR py -3.10 -m venv venv)
venv\Scripts\activate (for powershell: venv\Scripts\Activate.ps1)
python -m pip install -r requirements.txt
py -3.10 your_file.py (OR python your_file.py)



Torch usage for python 3.10
pip install --no-cache-dir torch==2.6.0+cpu torchvision==0.21.0+cpu torchaudio --index-url https://download.pytorch.org/whl/cpu





Torch use cpu( python 3.12, we dont need this)
In requirements file
torch==2.6.0+cpu
torchvision==0.21.0+cpu
torchaudio
numpy<2
TMPDIR=$HOME pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -r requirements.txt
