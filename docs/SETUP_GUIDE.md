# Setup Guide — FaceVerification (Windows)

## Prerequisites

- Windows 10 or 11
- Python 3.10 (confirmed installed at `C:\Users\tojin.s\AppData\Local\Programs\Python\Python310\python.exe`)
- Git
- A webcam
- A reference photo of the person to verify (JPG or PNG)

---

## Step 1: Create the development branch

This should already be done. Verify with:

```powershell
git branch
```

Expected output: `* feature/faceverification-modernization`

---

## Step 2: Create and activate a virtual environment

```powershell
cd "C:\Users\tojin.s\Music\Projects\FaceVerification"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If you see a permissions error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then retry the activate command.

After activation your prompt should show `(.venv)`.

---

## Step 3: Uninstall headless OpenCV and install correct dependencies

The global environment has `opencv-python-headless` installed, which cannot display GUI windows. Inside the venv, install the correct packages:

```powershell
pip install --upgrade pip
pip install opencv-python==4.10.0.84
pip install numpy==1.26.4
pip install tensorflow==2.15.0
pip install tf-keras==2.15.1
pip install deepface==0.0.93
pip install pillow==10.4.0
pip install python-dotenv==1.0.1
```

**Important:** Install in this order. numpy must be pinned to 1.26.4 — TensorFlow 2.15 is incompatible with numpy 2.x.

Verify installation:
```powershell
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import tensorflow as tf; print('TF:', tf.__version__)"
python -c "from deepface import DeepFace; print('DeepFace OK')"
```

---

## Step 4: Set up the reference image

Create a `.env` file (do not commit this file):

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set the path to your reference image:

```
REFERENCE_IMAGE_PATH=C:\path\to\your\reference_photo.jpg
```

The reference photo should:
- Show one face clearly
- Be well-lit and in focus
- Be the same person you will verify at runtime

---

## Step 5: Run the application

```powershell
python face-recognition.py
```

A window named "video" should appear showing the webcam feed.
- If a face matches the reference image → green "MATCH" text appears
- If no match → red "NOT A MATCH" text appears
- Press `q` to exit

---

## Troubleshooting

### "Error: Could not open camera"
- Check that your webcam is connected and not in use by another application
- Try a different camera index: change `VideoCapture(0)` to `VideoCapture(1)` in the script

### "reference_img is None / Error loading reference image"
- Verify the path in `.env` is correct
- Verify the file exists: `Test-Path "C:\path\to\your\photo.jpg"`
- Use a JPEG or PNG file

### cv2.imshow fails / no window appears
- Ensure you installed `opencv-python` (not `opencv-python-headless`)
- Check: `pip show opencv-python` — should show version 4.10.0.84

### TensorFlow import error with numpy
- You likely have numpy 2.x installed globally. Inside the venv:
  ```powershell
  pip install numpy==1.26.4 --force-reinstall
  ```

### DeepFace downloads models on first run
- First run will download ~580 MB (VGG-Face model) from the internet
- This is normal — subsequent runs use the cached model at `~/.deepface/weights/`
- Ensure you have an internet connection on first run

### Application freezes / very slow
- This is expected with VGG-Face on CPU — each inference takes 0.3–2 seconds
- This will be fixed in Version 1.2 with threaded inference

---

## File structure after setup

```
FaceVerification/
├── .venv/                  # Virtual environment (gitignored)
├── .env                    # Your local config (gitignored)
├── .env.example            # Template (committed)
├── .gitignore
├── requirements.txt
├── face-recognition.py
├── README.md
└── docs/
```
