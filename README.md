# Construction Helmet Detection Demo

This is a university demonstration web application for detecting:

- `helmet`: a worker wearing a safety helmet
- `no_helmet`: a worker without a safety helmet

The app uses a trained Ultralytics YOLO model and accepts a JPG or PNG image.
It is not a replacement for human supervision or formal construction-safety
procedures.

## Run locally

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open `http://localhost:8501` in a browser.

## Public demo deployment

1. Create a GitHub repository and push this project.
2. Confirm that the trained model is included at
   `runs/construction_helmet_yolo26nExtra/weights/best.pt`.
3. In Streamlit Community Cloud, select the GitHub repository and set
   `streamlit_app.py` as the entrypoint file.
4. Deploy and test the public URL with several non-sensitive images.

The deployment automatically uses an NVIDIA GPU when one is available and
otherwise uses CPU, which is expected for most public demo hosts.

## Included deployment files

- `requirements.txt` lists the Python dependencies for the cloud build.
- `.streamlit/config.toml` limits uploaded files to 10 MB.
- `.gitignore` excludes datasets, environments, and training artifacts while
  retaining the one `best.pt` model file required by the public demo.
