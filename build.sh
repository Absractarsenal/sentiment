#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting build script..."

# 1. Install all Python dependencies from requirements.txt
echo "Installing Python dependencies..."
pip install -r requirements.txt

# 2. Download NLTK stopwords data
echo "Downloading NLTK stopwords data..."
python -c "import nltk; nltk.download('stopwords')"

# 3. Create the 'models' directory if it doesn't exist
echo "Creating models directory..."
mkdir -p models

# 4. Download models from Google Drive
# This runs a small Python script to check for and download models
echo "Checking for and downloading models..."
python -c "
import os
import requests
import sys
from tqdm import tqdm

MODEL_URLS = {
    'FINBERT_FINAL.BIN': 'https://drive.google.com/uc?export=download&id=YOUR_FINBERT_ID_HERE',
    'SVM_FINAL.PKL': 'https://drive.google.com/uc?export=download&id=YOUR_SVM_ID_HERE',
    'TFIDF_VECTORIZER_FINAL.PKL': 'https://drive.google.com/uc?export=download&id=YOUR_TFIDF_ID_HERE'
}

MODEL_DIR = 'models'

def download_file(url, dest_path):
    print(f'Downloading {os.path.basename(dest_path)}...')
    session = requests.Session()
    response = session.get(url, stream=True)
    
    if 'download_warning' in response.url:
        print('Bypassing Google Drive download warning...')
        token = response.url.split('id=')[1].split('&')[0]
        params = {'id': token, 'confirm': 't'}
        response = session.get(url, params=params, stream=True)

    try:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(dest_path, 'wb') as f, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(dest_path)
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f'Successfully downloaded {os.path.basename(dest_path)}.')
    except requests.exceptions.RequestException as e:
        print(f'Error downloading {url}: {e}', file=sys.stderr)
        if os.path.exists(dest_path):
            os.remove(dest_path)
        sys.exit(1)

for model_name, url in MODEL_URLS.items():
    dest_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(dest_path):
        download_file(url, dest_path)
"
echo "Build script finished successfully."