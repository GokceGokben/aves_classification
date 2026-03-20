# Aves Classification

Bird species classification project with:
- FastAPI web service for browser/API use
- CLIP-based zero-shot inference pipeline

## Requirements
- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Run as Web/API

```bash
python app.py
```

Then open `http://localhost:8000`.

## Training

```bash
python main.py
```

## Docker (Web/API mode)

Build image:

```bash
docker build -t aves-classification .
```

Run container:

```bash
docker run --rm -p 8000:8000 aves-classification
```

Open `http://localhost:8000`.

## Project Structure
- `app.py`: FastAPI entrypoint
- `src/inference.py`: prediction pipeline
- `main.py`: training entrypoint
- `data/`: datasets and cached embeddings

## Dataset Credits
- **NABirds** dataset is used for supervised training assets under `data/nabirds`.
- Bird-name coverage list is built from public checklist sources used in `data/download_world_birds.py`:
	- `nicrie/bird-species` (GitHub CSV mirror)
	- `weecology/bird-phylogeny` (IOC mirror CSV)
- The script also includes a built-in fallback species list for reliability.

## Git / Privacy Notes
- Sensitive and heavy files are excluded via `.gitignore` before pushing to GitHub.
- This includes datasets, model checkpoints/weights, virtual env files, cache files, and temp test images.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Additional Note 
Model can make mistakes. Please verify critical results manually.
