# Skin Lesion Classification — Streamlit Research Demo

A mobile-friendly Streamlit web interface using the project's MobileNetV3Small TensorFlow Lite model.

## 1. Add the converted model

Copy this file from the existing project:

`D:\Skin_cancer_Project\models\mobilenetv3small_float32.tflite`

into this folder:

`models\mobilenetv3small_float32.tflite`

The trained model itself is intentionally not bundled with this starter archive.

## 2. Run locally on Windows

Use Python 3.12 and run in PowerShell from this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## 3. Deploy with Streamlit Community Cloud

1. Create a dedicated GitHub repository containing only this app, `requirements.txt`, `README.md`, `.gitignore`, and the model in `models/`.
2. Do not commit your HAM10000/ISIC image dataset, personal images, `.keras` training checkpoints, virtual environment, or unrelated project files.
3. Sign in at https://share.streamlit.io/ with GitHub access.
4. Choose **Create app**, select the repository and branch, and set the entrypoint to `app.py`.
5. Select Python 3.12 in Advanced settings if prompted, then deploy.
6. Open the resulting Streamlit URL on your phone.

## Important deployment and privacy notes

- This is a research demonstration, not a clinically validated diagnostic tool.
- The app performs prediction on the hosting server. Images selected in the browser are transmitted to that server for inference. Do not upload identifiable or sensitive images unless you understand and accept this.
- The code does not intentionally save uploaded images to disk, but hosting infrastructure processes requests and may have its own operational logging and policies.
- Check that preprocessing and class order match training before using the result for research reporting.
- The configured preprocessing resizes to 224 × 224 RGB and scales pixels to [0, 1], matching the project pipeline notes. Verify against the actual training loader before treating results as final.
