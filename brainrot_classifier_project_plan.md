# Brainrot Classifier — Project Plan
### Multimodal Video Understanding with Vision Transformer + Whisper

---

## Project Summary

A multimodal deep learning system that classifies short-form videos (TikTok/Reels style) as **brainrot** or **not brainrot**. The model samples frames from each video, extracts visual features using a fine-tuned Vision Transformer (ViT), transcribes audio using OpenAI Whisper, and fuses both signals through a learned fusion layer to output a final score and label.

**Sprint 4 adds a FastAPI backend + React frontend** where you paste a video URL and get a brainrot verdict.

---

## What You Will Learn

| Skill | Where |
|---|---|
| PyTorch training loops from scratch | Sprint 2 |
| Vision Transformer (ViT) — attention on image patches | Sprint 2 |
| Multimodal fusion (vision + audio/text) | Sprint 3 |
| Whisper for audio transcription | Sprint 3 |
| FastAPI — file upload, inference endpoint | Sprint 4 |
| React frontend — video upload + result display | Sprint 4 |
| Dataset curation and labelling pipeline | Sprint 1 |

---

## Hardware

| Task | Where to run |
|---|---|
| Dataset download & preprocessing | Main PC |
| ViT fine-tuning (CPU, ~1–2h per epoch) | Main PC overnight |
| Whisper transcription of dataset | Main PC |
| Inference serving (post-training) | Raspberry Pi 5 (8GB) |
| FastAPI + React in production | Raspberry Pi 5 |

> **Note:** You are not training ViT from scratch. You are fine-tuning `google/vit-base-patch16-224` which has pretrained weights. The heavy lifting is already done — you are teaching it to recognise brainrot-specific patterns on top of its existing visual understanding.

---

## Full Tech Stack

| Layer | Library / Tool |
|---|---|
| Video downloading | `yt-dlp` |
| Video frame extraction | `opencv-python` (cv2) |
| Audio extraction | `ffmpeg` (CLI via subprocess) |
| Audio transcription | `openai-whisper` |
| Vision model | `transformers` — `ViTForImageClassification` |
| Text encoding (for transcript) | `transformers` — `DistilBertModel` |
| Training | `PyTorch` + `torch.utils.data` |
| Experiment tracking | `matplotlib` + manual logs (keep it simple) |
| Inference API | `FastAPI` + `uvicorn` |
| Frontend | `React` + `Tailwind CSS` |
| Deployment | `Docker Compose` on Raspberry Pi 5 |

---

## Project Structure

```
brainrot-classifier/
├── data/
│   ├── raw/                  # downloaded .mp4 files
│   ├── frames/               # extracted frames per video
│   ├── transcripts/          # whisper output per video
│   └── labels.csv            # video_id, label (1=brainrot, 0=not)
├── src/
│   ├── data/
│   │   ├── downloader.py     # yt-dlp wrapper
│   │   ├── preprocessor.py   # frame extraction + audio split
│   │   └── dataset.py        # PyTorch Dataset class
│   ├── models/
│   │   ├── vision_encoder.py # ViT wrapper
│   │   ├── text_encoder.py   # DistilBERT wrapper for transcripts
│   │   └── fusion_model.py   # combined model
│   ├── train.py              # training loop
│   ├── evaluate.py           # metrics + confusion matrix
│   └── inference.py          # single video prediction
├── api/
│   └── main.py               # FastAPI app
├── frontend/                 # React app
├── notebooks/
│   └── 01_attention_demo.ipynb  # understand attention before using it
├── requirements.txt
├── .env
├── .gitignore
└── docker-compose.yml
```

---

## Data Sources

### Brainrot class (label = 1)
These subreddits self-label content as brainrot by nature:

| Source | URL | Notes |
|---|---|---|
| r/brainrot | reddit.com/r/brainrot | Primary source |
| r/shitposting | reddit.com/r/shitposting | High overlap |
| r/DeepFriedMemes | reddit.com/r/DeepFriedMemes | Visual chaos |
| r/okbuddyretard | reddit.com/r/okbuddyretard | Absurdist content |

### Not-brainrot class (label = 0)
Clean, coherent, informational or genuinely interesting content:

| Source | URL | Notes |
|---|---|---|
| r/educationalgifs | reddit.com/r/educationalgifs | Clear negative class |
| r/nextfuckinglevel | reddit.com/r/nextfuckinglevel | Skilled/impressive content |
| r/oddlyinteresting | reddit.com/r/oddlyinteresting | Calm, coherent |
| r/Documentaries (clips) | reddit.com/r/Documentaries | Structured content |

### Target dataset size
- **Minimum viable:** 200 videos per class (400 total) — enough to fine-tune
- **Good:** 400 per class (800 total) — more stable training
- Aim for 8–15 second clips. Ignore anything over 60 seconds.

---

## Sprint 0 — Environment Setup
**Time: Half a day. Do this before anything else.**

### Step 1 — Git + venv
```bash
mkdir brainrot-classifier && cd brainrot-classifier
git init

# Create .gitignore immediately
cat > .gitignore << EOF
.env
.venv/
__pycache__/
*.pyc
data/raw/
data/frames/
data/transcripts/
*.pt
*.pth
model_output/
EOF

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### Step 2 — Install dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets opencv-python-headless openai-whisper
pip install yt-dlp praw requests pandas numpy matplotlib scikit-learn
pip install fastapi uvicorn[standard] python-multipart python-dotenv
pip freeze > requirements.txt
```

### Step 3 — Install ffmpeg (system-level)
```bash
# Ubuntu / Raspberry Pi
sudo apt install ffmpeg -y

# Mac
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

### Step 4 — Reddit API credentials
1. Go to `https://www.reddit.com/prefs/apps`
2. Click **Create App** → choose **script**
3. Name it anything, redirect URI: `http://localhost:8080`
4. Copy your `client_id` and `client_secret`
5. Add to `.env`:
```
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
REDDIT_USER_AGENT=brainrot-classifier/1.0
```

### Step 5 — First commit
```bash
git add .
git commit -m "initial project scaffold"
```

---

## Sprint 1 — Dataset Collection & Preprocessing
**Goal: A clean, labelled dataset of video frames and transcripts ready for training.**
**Time: ~1 week**

---

### Task 1.1 — Reddit video scraper (`src/data/downloader.py`)

Use PRAW (Python Reddit API Wrapper) to find video posts, then yt-dlp to download them.

```python
import praw
import yt_dlp
import os
from dotenv import load_dotenv

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

def scrape_subreddit(subreddit_name: str, label: int, limit: int = 200):
    """
    Scrapes top video posts from a subreddit and downloads them.
    label: 1 = brainrot, 0 = not brainrot
    """
    subreddit = reddit.subreddit(subreddit_name)
    downloaded = []

    for post in subreddit.hot(limit=500):  # fetch more than needed, filter below
        # Only process video posts
        if not post.is_video:
            continue

        # Skip if too long (over 60 seconds)
        if hasattr(post, 'media') and post.media:
            duration = post.media.get('reddit_video', {}).get('duration', 999)
            if duration > 60:
                continue

        video_id = post.id
        out_path = f"data/raw/{video_id}.mp4"

        if os.path.exists(out_path):
            continue  # already downloaded

        try:
            ydl_opts = {
                'outtmpl': out_path,
                'format': 'mp4',
                'quiet': True,
                'max_filesize': 50 * 1024 * 1024,  # 50MB cap
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://reddit.com{post.permalink}"])

            downloaded.append({'video_id': video_id, 'label': label, 'source': subreddit_name})
            print(f"Downloaded {video_id} from r/{subreddit_name}")

            if len(downloaded) >= limit:
                break

        except Exception as e:
            print(f"Failed {video_id}: {e}")
            continue

    return downloaded
```

**Run it like this** in a script or notebook:
```python
from src.data.downloader import scrape_subreddit
import pandas as pd

results = []
results += scrape_subreddit("brainrot", label=1, limit=200)
results += scrape_subreddit("shitposting", label=1, limit=100)
results += scrape_subreddit("educationalgifs", label=0, limit=200)
results += scrape_subreddit("nextfuckinglevel", label=0, limit=100)

df = pd.DataFrame(results)
df.to_csv("data/labels.csv", index=False)
print(f"Total videos: {len(df)}")
```

> **Tip:** Run this overnight. Reddit rate limits are gentle but downloading 400 videos takes time. Check `data/labels.csv` in the morning.

---

### Task 1.2 — Frame extraction (`src/data/preprocessor.py`)

Sample exactly **N frames** uniformly across each video. This is your strategy for handling variable-length videos consistently.

```python
import cv2
import os
import numpy as np

def extract_frames(video_path: str, output_dir: str, n_frames: int = 8) -> list[str]:
    """
    Extracts n_frames evenly spaced frames from a video.
    Saves as JPGs. Returns list of saved frame paths.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"Warning: {video_path} has 0 frames, skipping.")
        return []

    # Evenly spaced frame indices
    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # Resize to 224x224 — ViT input size
        frame = cv2.resize(frame, (224, 224))
        frame_path = os.path.join(output_dir, f"frame_{i:02d}.jpg")
        cv2.imwrite(frame_path, frame)
        saved_paths.append(frame_path)

    cap.release()
    return saved_paths


def extract_audio(video_path: str, output_path: str) -> bool:
    """
    Extracts audio from video as 16kHz mono WAV (Whisper's required format).
    Uses ffmpeg via subprocess.
    """
    import subprocess
    cmd = [
        "ffmpeg", "-i", video_path,
        "-ac", "1",           # mono
        "-ar", "16000",       # 16kHz sample rate
        "-y",                 # overwrite if exists
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0
```

**Run preprocessing on your full dataset:**
```python
import pandas as pd
from src.data.preprocessor import extract_frames, extract_audio

df = pd.read_csv("data/labels.csv")

for _, row in df.iterrows():
    vid = row['video_id']
    video_path = f"data/raw/{vid}.mp4"
    frames_dir = f"data/frames/{vid}"
    audio_path = f"data/audio/{vid}.wav"

    if not os.path.exists(video_path):
        continue

    extract_frames(video_path, frames_dir, n_frames=8)
    extract_audio(video_path, audio_path)

print("Preprocessing complete.")
```

---

### Task 1.3 — Audio transcription with Whisper

Whisper runs locally, no API key needed. Use the `tiny` or `base` model — fast enough on CPU, accurate enough for short clips.

```python
import whisper
import os
import json

model = whisper.load_model("base")  # ~145MB, good balance of speed/accuracy

def transcribe_all(audio_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(audio_dir):
        if not fname.endswith(".wav"):
            continue

        video_id = fname.replace(".wav", "")
        out_path = f"{output_dir}/{video_id}.json"

        if os.path.exists(out_path):
            continue  # already transcribed

        audio_path = os.path.join(audio_dir, fname)

        try:
            result = model.transcribe(audio_path, language="en")
            with open(out_path, "w") as f:
                json.dump({"text": result["text"]}, f)
            print(f"Transcribed {video_id}: {result['text'][:60]}...")
        except Exception as e:
            print(f"Failed {video_id}: {e}")
            # Save empty transcript so we don't retry forever
            with open(out_path, "w") as f:
                json.dump({"text": ""}, f)

transcribe_all("data/audio", "data/transcripts")
```

> **Note:** Some brainrot videos have no speech, just music or sound effects. That's fine — an empty transcript is valid input. The model will learn to rely more on visual features for those.

---

### Task 1.4 — PyTorch Dataset class (`src/data/dataset.py`)

This class is the bridge between your files on disk and PyTorch's training loop. Understand every line.

```python
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import json
import os

# ViT expects images normalized to ImageNet stats
VIT_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class BrainrotDataset(Dataset):
    def __init__(self, labels_csv: str, frames_dir: str, transcripts_dir: str,
                 n_frames: int = 8, max_text_len: int = 64, tokenizer=None):
        self.df = pd.read_csv(labels_csv)
        self.frames_dir = frames_dir
        self.transcripts_dir = transcripts_dir
        self.n_frames = n_frames
        self.max_text_len = max_text_len
        self.tokenizer = tokenizer  # DistilBERT tokenizer, passed in from outside

        # Filter to only rows where we have frames
        self.df = self.df[
            self.df['video_id'].apply(
                lambda v: os.path.exists(os.path.join(frames_dir, v))
            )
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = row['video_id']
        label = int(row['label'])

        # --- Load frames ---
        frames = []
        frame_dir = os.path.join(self.frames_dir, video_id)
        frame_files = sorted(os.listdir(frame_dir))[:self.n_frames]

        for fname in frame_files:
            img = Image.open(os.path.join(frame_dir, fname)).convert("RGB")
            frames.append(VIT_TRANSFORMS(img))

        # Pad with zeros if fewer than n_frames (short video)
        while len(frames) < self.n_frames:
            frames.append(torch.zeros(3, 224, 224))

        frames_tensor = torch.stack(frames)  # shape: (n_frames, 3, 224, 224)

        # --- Load transcript ---
        transcript_path = os.path.join(self.transcripts_dir, f"{video_id}.json")
        if os.path.exists(transcript_path):
            with open(transcript_path) as f:
                text = json.load(f).get("text", "")
        else:
            text = ""

        # Tokenize for DistilBERT
        if self.tokenizer and text:
            encoded = self.tokenizer(
                text,
                max_length=self.max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
        else:
            input_ids = torch.zeros(self.max_text_len, dtype=torch.long)
            attention_mask = torch.zeros(self.max_text_len, dtype=torch.long)

        return {
            "frames": frames_tensor,           # (8, 3, 224, 224)
            "input_ids": input_ids,             # (64,)
            "attention_mask": attention_mask,   # (64,)
            "label": torch.tensor(label, dtype=torch.long)
        }
```

---

### Task 1.5 — Verify the dataset

Before training anything, sanity check the data:

```python
from transformers import DistilBertTokenizer
from src.data.dataset import BrainrotDataset
from torch.utils.data import DataLoader

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

dataset = BrainrotDataset(
    labels_csv="data/labels.csv",
    frames_dir="data/frames",
    transcripts_dir="data/transcripts",
    tokenizer=tokenizer
)

print(f"Dataset size: {len(dataset)}")
print(f"Sample keys: {dataset[0].keys()}")
print(f"Frames shape: {dataset[0]['frames'].shape}")   # should be (8, 3, 224, 224)
print(f"Input IDs shape: {dataset[0]['input_ids'].shape}")  # should be (64,)
print(f"Label: {dataset[0]['label']}")

# Check class balance
import pandas as pd
df = pd.read_csv("data/labels.csv")
print(df['label'].value_counts())  # aim for roughly 50/50
```

### Sprint 1 Deliverable

Running `python scripts/build_dataset.py` produces:
- `data/frames/{video_id}/frame_00.jpg ... frame_07.jpg` for every video
- `data/transcripts/{video_id}.json` for every video
- `data/labels.csv` with ~400–800 labelled rows
- `BrainrotDataset[0]` returns correct tensor shapes with no errors

---

## Sprint 2 — Vision Model & Training Loop
**Goal: A fine-tuned ViT that classifies videos by visual content alone. Understand what you're doing.**
**Time: ~1 week**

---

### Task 2.1 — Understand attention before touching the model (mandatory notebook)

Open `notebooks/01_attention_demo.ipynb` and implement this yourself:

```python
import torch
import torch.nn.functional as F

def self_attention(Q, K, V):
    """
    Q, K, V: query, key, value matrices
    In ViT, each image patch becomes one token.
    The model learns which patches to 'pay attention to' when classifying.
    """
    d_k = Q.shape[-1]  # dimension of each key/query vector
    
    # Compute attention scores: how similar is each query to each key?
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    
    # Convert to probabilities (each row sums to 1)
    weights = F.softmax(scores, dim=-1)
    
    # Weighted sum of values
    output = torch.matmul(weights, V)
    return output, weights

# Toy example: 4 patches, 8-dimensional embeddings
n_patches = 4
d_model = 8
Q = torch.randn(n_patches, d_model)
K = torch.randn(n_patches, d_model)
V = torch.randn(n_patches, d_model)

output, weights = self_attention(Q, K, V)
print("Attention weights (each row sums to 1):")
print(weights.detach().numpy().round(3))
# Each row shows how much patch i attends to every other patch
```

**Why this matters for your project:** ViT splits each 224×224 image into 14×14 = 196 patches (16px each). Each patch becomes a token. The attention mechanism learns that for brainrot classification, certain patches (chaotic overlays, meme text, rapidly changing regions) matter more than others. You're not implementing this yourself — but understanding it means you can explain your model in an interview.

---

### Task 2.2 — Vision encoder (`src/models/vision_encoder.py`)

```python
import torch
import torch.nn as nn
from transformers import ViTModel

class FrameEncoder(nn.Module):
    """
    Encodes a single video frame using a pretrained Vision Transformer.
    Returns a fixed-size embedding vector per frame.
    """
    def __init__(self, pretrained_name: str = "google/vit-base-patch16-224",
                 output_dim: int = 256, freeze_backbone: bool = True):
        super().__init__()
        self.vit = ViTModel.from_pretrained(pretrained_name)
        vit_hidden_size = self.vit.config.hidden_size  # 768 for vit-base

        # Freeze ViT weights initially — we only train the projection head
        # This is critical for CPU training: fewer parameters = faster
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        # Learned projection: 768 → 256
        self.projection = nn.Sequential(
            nn.Linear(vit_hidden_size, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: (batch_size, 3, 224, 224)
        returns: (batch_size, output_dim)
        """
        outputs = self.vit(pixel_values=pixel_values)
        # [CLS] token output — summary of the whole image
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        return self.projection(cls_output)               # (batch, 256)


class VideoEncoder(nn.Module):
    """
    Encodes all frames of a video and aggregates them into one vector.
    Strategy: encode each frame independently, then mean-pool across frames.
    """
    def __init__(self, frame_encoder: FrameEncoder):
        super().__init__()
        self.frame_encoder = frame_encoder

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (batch_size, n_frames, 3, 224, 224)
        returns: (batch_size, output_dim)
        """
        batch_size, n_frames, C, H, W = frames.shape

        # Reshape to process all frames at once: (batch*n_frames, 3, 224, 224)
        frames_flat = frames.view(batch_size * n_frames, C, H, W)
        frame_embeddings = self.frame_encoder(frames_flat)  # (batch*n_frames, 256)

        # Reshape back and average across frames
        frame_embeddings = frame_embeddings.view(batch_size, n_frames, -1)
        video_embedding = frame_embeddings.mean(dim=1)  # (batch, 256)
        return video_embedding
```

> **Why freeze the backbone?** `vit-base-patch16-224` has 86M parameters. Training all of them on CPU would take days. By freezing and only training the projection head + fusion layer (a few thousand parameters), each epoch takes minutes. After a few epochs you can unfreeze the last 2 transformer blocks for fine-tuning if you want to push accuracy higher.

---

### Task 2.3 — Text encoder (`src/models/text_encoder.py`)

```python
import torch
import torch.nn as nn
from transformers import DistilBertModel

class TranscriptEncoder(nn.Module):
    """
    Encodes the video's audio transcript using DistilBERT.
    Returns a fixed-size embedding vector.
    """
    def __init__(self, pretrained_name: str = "distilbert-base-uncased",
                 output_dim: int = 256, freeze_backbone: bool = True):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(pretrained_name)

        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False

        bert_hidden_size = self.bert.config.hidden_size  # 768 for distilbert-base

        self.projection = nn.Sequential(
            nn.Linear(bert_hidden_size, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        returns: (batch_size, output_dim)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.projection(cls_output)
```

---

### Task 2.4 — Fusion model (`src/models/fusion_model.py`)

This is the core of the multimodal architecture. You combine the visual and text representations, then classify.

```python
import torch
import torch.nn as nn
from src.models.vision_encoder import FrameEncoder, VideoEncoder
from src.models.text_encoder import TranscriptEncoder

class BrainrotClassifier(nn.Module):
    """
    Multimodal brainrot classifier.
    Fuses visual features from ViT with transcript features from DistilBERT.
    """
    def __init__(self, visual_dim: int = 256, text_dim: int = 256,
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        # Encoders
        frame_encoder = FrameEncoder(output_dim=visual_dim, freeze_backbone=True)
        self.video_encoder = VideoEncoder(frame_encoder)
        self.text_encoder = TranscriptEncoder(output_dim=text_dim, freeze_backbone=True)

        # Fusion: concatenate visual + text → classify
        # Input dim = visual_dim + text_dim = 512
        self.classifier = nn.Sequential(
            nn.Linear(visual_dim + text_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, frames, input_ids, attention_mask):
        visual_features = self.video_encoder(frames)                        # (batch, 256)
        text_features = self.text_encoder(input_ids, attention_mask)        # (batch, 256)
        combined = torch.cat([visual_features, text_features], dim=1)       # (batch, 512)
        logits = self.classifier(combined)                                   # (batch, 2)
        return logits
```

---

### Task 2.5 — Training loop (`src/train.py`)

Write this yourself — do not use HuggingFace Trainer here. Understanding the raw PyTorch loop is the point.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import DistilBertTokenizer
from src.data.dataset import BrainrotDataset
from src.models.fusion_model import BrainrotClassifier
import os

# --- Config ---
EPOCHS = 10
BATCH_SIZE = 8      # keep low for CPU
LR = 1e-3
N_FRAMES = 8
CHECKPOINT_DIR = "model_output"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Data ---
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
dataset = BrainrotDataset("data/labels.csv", "data/frames",
                           "data/transcripts", tokenizer=tokenizer)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# --- Model ---
model = BrainrotClassifier()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)
criterion = nn.CrossEntropyLoss()

# --- Training loop ---
best_val_acc = 0.0
history = {"train_loss": [], "val_loss": [], "val_acc": []}

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        frames = batch["frames"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        optimizer.zero_grad()
        logits = model(frames, input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch["frames"], batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["label"])
            val_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == batch["label"]).sum().item()
            total += len(batch["label"])

    val_acc = correct / total
    avg_val_loss = val_loss / len(val_loader)

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pt")
        print(f"  → Saved new best model (val_acc={val_acc:.4f})")

print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
```

---

### Task 2.6 — Evaluation (`src/evaluate.py`)

```python
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch["frames"], batch["input_ids"], batch["attention_mask"])
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(batch["label"].tolist())

    print(classification_report(all_labels, all_preds,
                                  target_names=["not_brainrot", "brainrot"]))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=["not_brainrot", "brainrot"],
                yticklabels=["not_brainrot", "brainrot"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("model_output/confusion_matrix.png")
    print("Saved confusion_matrix.png")
```

**Target accuracy:** 75%+ is good for a first pass. 80%+ is solid. If you're below 70%, the most common fixes are class imbalance (check `labels.csv` value counts) and too few training samples.

### Sprint 2 Deliverable

- `model_output/best_model.pt` — saved weights
- Training loss curve going down across epochs
- Val accuracy ≥ 75% on held-out test set
- Confusion matrix saved to `model_output/confusion_matrix.png`
- You can explain what ViT attention does in plain English

---

## Sprint 3 — Inference Pipeline & Optimisation
**Goal: A clean `inference.py` that takes a raw video file and returns a prediction. Optimise for Pi.**
**Time: ~4–5 days**

---

### Task 3.1 — Single-video inference (`src/inference.py`)

```python
import torch
import tempfile
import os
from transformers import DistilBertTokenizer
from src.models.fusion_model import BrainrotClassifier
from src.data.preprocessor import extract_frames, extract_audio
from src.data.dataset import VIT_TRANSFORMS
from PIL import Image
import whisper
import json

class BrainrotInference:
    def __init__(self, model_path: str = "model_output/best_model.pt"):
        self.device = torch.device("cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.whisper_model = whisper.load_model("base")

        self.model = BrainrotClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.labels = {0: "not_brainrot", 1: "brainrot"}

    def predict(self, video_path: str) -> dict:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract frames
            frames_dir = os.path.join(tmpdir, "frames")
            frame_paths = extract_frames(video_path, frames_dir, n_frames=8)

            # Load and transform frames
            frames = []
            for fp in frame_paths:
                img = Image.open(fp).convert("RGB")
                frames.append(VIT_TRANSFORMS(img))
            while len(frames) < 8:
                frames.append(torch.zeros(3, 224, 224))
            frames_tensor = torch.stack(frames).unsqueeze(0)  # (1, 8, 3, 224, 224)

            # Extract and transcribe audio
            audio_path = os.path.join(tmpdir, "audio.wav")
            extract_audio(video_path, audio_path)
            result = self.whisper_model.transcribe(audio_path, language="en")
            transcript = result.get("text", "")

            # Tokenize transcript
            encoded = self.tokenizer(
                transcript, max_length=64, padding="max_length",
                truncation=True, return_tensors="pt"
            )

        # Run inference
        with torch.no_grad():
            logits = self.model(
                frames_tensor,
                encoded["input_ids"],
                encoded["attention_mask"]
            )
            probs = torch.softmax(logits, dim=1).squeeze()
            pred_class = logits.argmax(dim=1).item()

        return {
            "label": self.labels[pred_class],
            "confidence": round(probs[pred_class].item(), 3),
            "scores": {
                "not_brainrot": round(probs[0].item(), 3),
                "brainrot": round(probs[1].item(), 3)
            },
            "transcript": transcript[:200]  # first 200 chars for display
        }
```

**Test it:**
```python
from src.inference import BrainrotInference

predictor = BrainrotInference()
result = predictor.predict("data/raw/some_video_id.mp4")
print(result)
# Expected output:
# {'label': 'brainrot', 'confidence': 0.87, 'scores': {...}, 'transcript': '...'}
```

---

### Task 3.2 — Copy model to Raspberry Pi

```bash
# From your main PC
scp model_output/best_model.pt pi@raspberrypi.local:/home/pi/brainrot-classifier/model_output/
scp requirements.txt pi@raspberrypi.local:/home/pi/brainrot-classifier/

# On Pi
cd /home/pi/brainrot-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

### Task 3.3 — Benchmark inference speed on Pi

```python
import time
from src.inference import BrainrotInference

predictor = BrainrotInference()
video_path = "data/raw/test_video.mp4"

# Warm-up run
predictor.predict(video_path)

# Timed runs
times = []
for _ in range(5):
    start = time.time()
    predictor.predict(video_path)
    times.append(time.time() - start)

avg = sum(times) / len(times)
print(f"Average inference time: {avg:.2f}s")
```

Expected on Pi 5: **8–15 seconds** per video (Whisper is the bottleneck). This is fine for a web app where users upload and wait for a result.

If it's too slow, switch Whisper to `tiny` model (faster, slightly less accurate):
```python
self.whisper_model = whisper.load_model("tiny")  # ~75MB, ~2x faster
```

### Sprint 3 Deliverable

- `predictor.predict("path/to/video.mp4")` returns a result dict in under 15 seconds on Pi
- Inference works correctly on a sample of 10 test videos
- Model weights are on the Pi

---

## Sprint 4 — FastAPI Backend + React Frontend
**Goal: Paste a local video file, get a brainrot score. Deployed on Pi.**
**Time: ~1 week**

---

### Task 4.1 — FastAPI app (`api/main.py`)

Key concepts you'll use that are new to you: **file upload**, **background tasks**, **job status polling**.

```python
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
import tempfile
import os
import shutil
from src.inference import BrainrotInference

app = FastAPI(title="Brainrot Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (fine for a demo project)
jobs: dict = {}
predictor = BrainrotInference()


def run_inference(job_id: str, video_path: str):
    """Runs in background — updates job store when done."""
    try:
        jobs[job_id]["status"] = "processing"
        result = predictor.predict(video_path)
        jobs[job_id] = {"status": "done", "result": result}
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}
    finally:
        # Clean up temp file
        if os.path.exists(video_path):
            os.remove(video_path)


@app.post("/api/classify")
async def classify_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Accept a video file upload, start async inference, return job ID."""
    if not file.filename.endswith((".mp4", ".mov", ".webm")):
        raise HTTPException(400, "Only .mp4, .mov, .webm files accepted")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued"}

    # Save uploaded file to temp location
    tmp_path = f"/tmp/{job_id}.mp4"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run inference in background (non-blocking)
    background_tasks.add_task(run_inference, job_id, tmp_path)

    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    """Poll this endpoint to check if inference is done."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": predictor is not None}
```

---

### Task 4.2 — React frontend

Keep it simple. One page: upload video → loading state → result card.

Key components:
- **VideoUpload** — drag-and-drop or file picker, sends to `/api/classify`
- **PollingLoader** — polls `/api/job/{id}` every 2 seconds until `status === "done"`
- **ResultCard** — shows label, confidence bar, transcript snippet

The result card for a brainrot verdict should look deliberately chaotic (red, flashing, Comic Sans if you're feeling it). For not-brainrot: calm, clean, green. Make the UI match the content — interviewers remember projects that have personality.

---

### Task 4.3 — Docker Compose

```yaml
version: "3.9"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./model_output:/app/model_output
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - api
    restart: unless-stopped
```

### Sprint 4 Deliverable

- Upload a video at `http://your-pi-url.com` → get a result
- Works end-to-end on Raspberry Pi 5
- Clean README with architecture diagram, demo screenshots, and setup instructions

---

## Final CV Bullet (draft)

> **Brainrot Classifier** — Built a multimodal video classification system combining a fine-tuned Vision Transformer (ViT) for frame-level visual features and DistilBERT for Whisper-transcribed audio, fused through a learned MLP to classify short-form videos. Wrote the full PyTorch training loop from scratch including data pipeline, frame sampling, and multimodal dataset class. Served via a FastAPI async inference API with React frontend, deployed on Raspberry Pi 5 via Docker Compose.
> *Tools: Python, PyTorch, HuggingFace Transformers (ViT, DistilBERT), Whisper, FastAPI, React, OpenCV, Docker*

---

## Context Window Recovery Prompt

If this plan falls out of context, paste this to resume:

> "I'm building a Brainrot Classifier — a multimodal deep learning project that classifies short videos as brainrot or not using a Vision Transformer (ViT) for frames and DistilBERT for Whisper transcripts, fused through an MLP. 4 sprints: Sprint 0 = env setup, Sprint 1 = dataset collection (Reddit via PRAW + yt-dlp, frame extraction with OpenCV, Whisper transcription), Sprint 2 = ViT + DistilBERT encoders + PyTorch training loop, Sprint 3 = inference pipeline + Pi deployment, Sprint 4 = FastAPI + React frontend + Docker. I am currently on Sprint [X], Task [X.Y]."
