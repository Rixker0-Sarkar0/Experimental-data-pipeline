import cv2
import chromadb
import torch
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os

# -----------------------
# 1. Initialize GPU model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()
model.eval().to(device)

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# -----------------------
# 2. Streaming frame embeddings
# -----------------------
def stream_video_embeddings(video_path, interval=1, batch_size=32):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    buffer_frames, buffer_ts = [], []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % (fps * interval) == 0:
            buffer_frames.append(frame)
            buffer_ts.append(idx / fps)
        idx += 1
        if len(buffer_frames) == batch_size:
            embeddings = []
            with torch.no_grad():
                batch_tensor = torch.stack([transform(Image.fromarray(f)) for f in buffer_frames]).to(device)
                batch_emb = model(batch_tensor).cpu().numpy()
                embeddings.append(batch_emb)
            embeddings = np.vstack(embeddings)
            for e, ts in zip(embeddings, buffer_ts):
                yield e, ts
            buffer_frames, buffer_ts = [], []
    if buffer_frames:
        embeddings = []
        with torch.no_grad():
            batch_tensor = torch.stack([transform(Image.fromarray(f)) for f in buffer_frames]).to(device)
            batch_emb = model(batch_tensor).cpu().numpy()
            embeddings.append(batch_emb)
        embeddings = np.vstack(embeddings)
        for e, ts in zip(embeddings, buffer_ts):
            yield e, ts
    cap.release()

# -----------------------
# 3. Initialize ChromaDB
# -----------------------
client = chromadb.Client()
collection_name = "video_reference"
try:
    collection = client.get_collection(collection_name)
except:
    collection = client.create_collection(collection_name)

# -----------------------
# 4. Load open datasets (anchor)
# -----------------------
def build_reference_anchor(open_datasets, cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    ref_id_counter = 0
    for dataset_name, video_path in open_datasets.items():
        cache_file = os.path.join(cache_dir, f"{dataset_name}_embeddings.npy")
        ts_file = os.path.join(cache_dir, f"{dataset_name}_timestamps.npy")
        if os.path.exists(cache_file) and os.path.exists(ts_file):
            vecs = np.load(cache_file)
            ts = np.load(ts_file)
        else:
            vecs_list, ts = [], []
            for e, t in stream_video_embeddings(video_path):
                vecs_list.append(e)
                ts.append(t)
            vecs = np.vstack(vecs_list)
            np.save(cache_file, vecs)
            np.save(ts_file, ts)
        ids = [f"ref_{ref_id_counter + i}" for i in range(len(vecs))]
        meta = [{"ref_dataset": dataset_name.split("_")[0],
                 "ref_video": dataset_name,
                 "timestamp": t} for t in ts]
        collection.add(embeddings=vecs.tolist(), ids=ids, metadatas=meta)
        ref_id_counter += len(vecs)

open_datasets = {
    "UCF101_v1": "ref_video1.mp4",
    "Kinetics_v2": "ref_video2.mp4"
}
build_reference_anchor(open_datasets)

# -----------------------
# 5. Adaptive threshold
# -----------------------
def compute_adaptive_threshold(collection, sample_size=1000, k=2.0):
    all_ids = collection.get()["ids"]
    sample_ids = np.random.choice(all_ids, min(sample_size, len(all_ids)), replace=False)
    distances = []
    for sid in sample_ids:
        vec = collection.get(ids=[sid])["embeddings"][0]
        q = collection.query(query_embeddings=[vec], n_results=2)
        distances.append(q["distances"][0][1])
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    threshold = mean_dist + k * std_dist
    print(f"Adaptive threshold: {threshold:.4f}")
    return threshold

threshold = compute_adaptive_threshold(collection, k=2.0)

# -----------------------
# 6. Stream comparison
# -----------------------
def stream_compare_video(video_path, video_id, collection, threshold):
    results = []
    for emb, ts in tqdm(stream_video_embeddings(video_path, interval=1), desc=f"Processing {video_id}"):
        q = collection.query(query_embeddings=[emb.tolist()], n_results=1)
        nearest_meta = q["metadatas"][0][0]
        distance = q["distances"][0][0]
        status = "matched" if distance <= threshold else "novel"
        results.append([
            video_id,
            ts,
            nearest_meta["ref_dataset"],
            nearest_meta["ref_video"],
            nearest_meta["timestamp"],
            distance,
            status
        ])
    return results

# -----------------------
# 7. Run streaming batch
# -----------------------
new_videos = {
    "videoA": "videoA.mp4",
    "videoB": "videoB.mp4"
}

all_results = []
for vid_id, path in new_videos.items():
    all_results.extend(stream_compare_video(path, vid_id, collection, threshold))

# -----------------------
# 8. Export CSV
# -----------------------
df = pd.DataFrame(all_results, columns=[
    "video_id","timestamp","ref_dataset","ref_video","ref_timestamp","distance","status"
])
df.to_csv("streaming_comparison_results.csv", index=False)
print("Streaming GPU comparison complete. CSV saved.")
