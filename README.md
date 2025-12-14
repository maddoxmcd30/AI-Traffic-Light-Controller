# AI-Traffic-Light-Controller


## Prerequisites (Windows + WSL)
This repo is meant to run using:
- Windows host
- WSL (Ubuntu recommended)


### 1) Install SUMO in WSL
In WSL:
```bash
sudo apt update
sudo apt install -y sumo sumo-tools sumo-doc
```


### 2) Python (WSL venv)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
You must set SUMO_HOME so Python can locate SUMO
```bash
export SUMO_HOME=/usr/share/sumo
```

### 3) View the trained model
```bash
python play_trained.py
```
### 3) Train a new model
This will replace the the model thats already there

```bash
python train_sumo.py
```
