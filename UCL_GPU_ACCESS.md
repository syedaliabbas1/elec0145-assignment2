# UCL GPU Access Guide
## ELEC0145 Assignment 2 — Lab 105 RTX 3090 Setup
**Username:** sabbas | **Target lab:** lab105 (RTX 3090, 24GB VRAM)

---

## Overview

| Item | Detail |
|------|--------|
| Target GPU | Nvidia RTX 3090 — 24GB RAM, 10,496 cores |
| Lab | lab105 |
| Access method | VS Code Remote SSH extension |
| SSH gateway | knuckles.cs.ucl.ac.uk |
| CS username | sabbas |

---

## Step 1 — Find an Available Machine in lab105

Lab 105 machines (all run Linux by default):

```
aylesbury-l   barnacle-l   brent-l     bufflehead-l   cackling-l
harlequin-l      crested-l    eider-l     gadwall-l      goosander-l
gressingham-l harlequin-l  mallard-l   mandarin-l     pintail-l
pocher-l      ruddy-l      scaup-l     scoter-l       shelduck-l
shoveler-l    smew-l       wigeon-l
```

**Check if a machine is online** by pinging it first (run this in your local terminal):

```bash
ping harlequin-l.cs.ucl.ac.uk
```

If you get a response — the machine is online. If not, try another one from the list.

---

## Step 2 — Configure VS Code SSH

**Open VS Code → F1 → "Remote-SSH: Open SSH Configuration File"**

Add this to your SSH config file (`~/.ssh/config` on Mac/Linux or `C:\Users\sabbas\.ssh\config` on Windows):

```
Host ucl-gpu
    HostName harlequin-l.cs.ucl.ac.uk
    User sabbas
    ProxyJump sabbas@knuckles.cs.ucl.ac.uk
```

> Change `harlequin-l` to whichever machine you pinged successfully in Step 1.

---

## Step 3 — Connect via VS Code

1. Press **F1** in VS Code
2. Type **"Remote-SSH: Connect to Host"**
3. Select **ucl-gpu**
4. Enter your UCL CS password when prompted (may be asked twice — once for the gateway, once for the GPU machine)
5. VS Code will connect and open a remote window

You are now inside the UCL GPU machine.

---

## Step 4 — Check GPU Availability

Once connected, open a terminal in VS Code (**Terminal → New Terminal**) and run:

```bash
nvidia-smi
```

This shows:
- GPU memory usage
- Processes currently running on the GPU
- GPU utilisation %

**If the GPU is occupied:** try a different machine from the lab105 list. Update your SSH config `HostName` to the new machine and reconnect.

**If the GPU is free:** you're good to go.

---

## Step 5 — Set Up Python Environment

Run these commands in the VS Code terminal **once** when you first connect:

**Add Python to your path:**
```bash
source /opt/Python/Python-3.10.1_Setup.csh
```

**Install required packages (use --user flag — required on UCL systems):**
```bash
pip install torch torchvision --user
pip install scikit-learn matplotlib seaborn --user
pip install jupyter ipykernel --user
```

**Verify GPU is detected by PyTorch:**
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output:
```
True
NVIDIA GeForce RTX 3090
```

---

## Step 6 — Transfer Your Dataset to UCL

From your **local terminal** (not the VS Code remote terminal), run:

```bash
scp -J sabbas@knuckles.cs.ucl.ac.uk -r "C:\Download\Studies\Robotics\Coursework 2\dataset\raw" sabbas@harlequin-l.cs.ucl.ac.uk:~/ELEC0145/dataset/
```

> Change `harlequin-l` to your target machine. Change the local path if needed.

Or use **VS Code drag and drop** — with Remote SSH connected, open the Explorer panel and drag your local dataset folder directly into the remote file explorer. Simpler and no command needed.

---

## Step 7 — Open Your Notebook

1. In VS Code remote window, open your project folder:
   **File → Open Folder → ~/ELEC0145/**

2. Open `task1_classifier.ipynb`

3. VS Code will prompt you to select a kernel — select the Python 3.10 environment

4. Run the first cell — confirm CUDA is available:
```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Should print RTX 3090
```

---

## Step 8 — Run Training

Everything runs normally in the notebook from here. Training on the RTX 3090 with 150 images will take **under 15 minutes**.

---

## Important Rules & Warnings

| Rule | Detail |
|------|--------|
| **Reboot window** | Machines reboot Monday and Thursday evenings 7:30pm–midnight — do NOT run training during these times |
| **Check GPU first** | Always run `nvidia-smi` before starting — someone else may be using it |
| **One GPU only** | On multi-GPU machines (Blaze), use only one card — not applicable for lab105 single-GPU machines |
| **--user flag** | Always use `pip install --user` — you don't have write access to the global filesystem |
| **Save checkpoints** | Although training is short, always save model weights to `~/ELEC0145/models/` so work is never lost |

---

## Switching Machines

If your current machine is busy or goes offline, update the `HostName` in your SSH config:

```
Host ucl-gpu
    HostName barnacle-l.cs.ucl.ac.uk   ← change this to a new machine
    User sabbas
    ProxyJump sabbas@knuckles.cs.ucl.ac.uk
```

Then reconnect via **F1 → Remote-SSH: Connect to Host → ucl-gpu**.

---

## Quick Reference Commands

```bash
# Check GPU availability
nvidia-smi

# Check which Python you're using
which python

# Add Python to path (run each session)
source /opt/Python/Python-3.10.1_Setup.csh

# Check PyTorch + CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check installed packages
pip list --user | grep torch

# Check disk space in your filespace
df -h ~
```

---

## Troubleshooting

| Problem | Solution |
|---------|---------|
| SSH connection refused | Machine is offline — ping another lab105 machine and update SSH config |
| GPU not detected by PyTorch | Run `source /opt/Python/Python-3.10.1_Setup.csh` first, then retry |
| `nvidia-smi` shows GPU occupied | Try a different machine |
| Permission denied on pip install | Make sure you're using `pip install --user` |
| VS Code kernel not found | Install ipykernel: `pip install ipykernel --user` |
| Connection drops mid-training | Use `tmux` or `nohup` to keep the job running if session disconnects |

---

*ELEC0145 Assignment 2 | UCL GPU Access Guide | Username: sabbas | Lab 105 RTX 3090*