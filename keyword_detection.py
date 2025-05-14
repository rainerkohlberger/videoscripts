import os
import subprocess
import torch
import clip
import json
import re
import csv
import uuid
from PIL import Image
from tqdm import tqdm
from collections import Counter
import xml.etree.ElementTree as ET
import sys
import argparse

# Settings
parser = argparse.ArgumentParser(description="Keyword tagger for QuickTime clips")
parser.add_argument("video_dir", nargs="?", default="videos", help="Directory containing .mov files")
args = parser.parse_args()
VIDEO_DIR = args.video_dir

FRAMES_DIR = "temp_frames"
KEYWORDS_PER_VIDEO = 3
FRAMES_PER_VIDEO = 20
csv_rows = []
fcpxml_assets = []
fcpxml_clips = []

LABELS = [
    "person", "man", "woman", "child", "group of people", "ghost", "zombie", "vampire", "monster",
    "witch", "killer", "demon", "creature", "corpse", "skeleton", "clown", "mask", "face", "eye",
    "scream", "fear", "terror", "shock", "panic", "blood", "wound", "injury", "violence", "death",
    "dead body", "gore", "possession", "decay", "insanity", "madness", "disturbance", "evil",
    "house", "haunted house", "forest", "graveyard", "cemetery", "morgue", "asylum", "hospital",
    "church", "basement", "attic", "abandoned building", "cabin", "woods", "room", "corridor",
    "tunnel", "hallway", "dark room", "cellar", "castle",
    "knife", "axe", "chainsaw", "cross", "mirror", "door", "window", "candle", "bloodstain",
    "doll", "television", "phone", "camera", "rope", "gun", "bat", "book", "ritual", "altar",
    "bat", "cat", "rat", "spider", "snake", "crow", "wolf", "insect",
    "darkness", "fog", "mist", "moonlight", "shadow", "red light", "blue light", "flickering light",
    "fire", "rain", "storm", "lightning", "night", "sunset", "eclipse",
    "street", "car", "bedroom", "kitchen", "bathroom", "closet", "garage", "cell", "symbol", "noise", "surveillance", "trap"
]

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
#model, preprocess = clip.load("ViT-B/32", device=device)
model, preprocess = clip.load("RN50", device=device)

def extract_frames(video_path, out_dir, max_frames=FRAMES_PER_VIDEO):
    os.makedirs(out_dir, exist_ok=True)
    out_pattern = os.path.join(out_dir, "frame_%03d.jpg")
    command = [
        "ffmpeg", "-i", video_path,
        "-vf", "fps=1",
        "-vframes", str(max_frames),
        "-q:v", "2",
        out_pattern,
        "-hide_banner", "-loglevel", "error"
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def classify_images(image_dir):
    keywords = []
    text_tokens = clip.tokenize([f"a photo of {label}" for label in LABELS]).to(device)

    for filename in sorted(os.listdir(image_dir)):
        if not filename.endswith(".jpg"):
            continue
        image_path = os.path.join(image_dir, filename)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            logits_per_image, _ = model(image, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        top_labels = [LABELS[i].strip().replace("scene", "").replace(" ", "-")
                      for i in probs.argsort()[-KEYWORDS_PER_VIDEO:][::-1]]
        keywords.extend(top_labels)

    return keywords

def create_contact_sheet(image_dir, output_path, thumb_size=(320, 180), images_per_row=5):
    images = [
        Image.open(os.path.join(image_dir, f)).resize(thumb_size)
        for f in sorted(os.listdir(image_dir)) if f.endswith(".jpg")
    ]
    if not images:
        return
    rows = (len(images) + images_per_row - 1) // images_per_row
    sheet = Image.new("RGB", (images_per_row * thumb_size[0], rows * thumb_size[1]))
    for i, img in enumerate(images):
        x = (i % images_per_row) * thumb_size[0]
        y = (i // images_per_row) * thumb_size[1]
        sheet.paste(img, (x, y))
    sheet.save(output_path)

def get_video_duration(filepath):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", filepath],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        seconds = float(result.stdout.strip())
        return f"{seconds:.2f}s"
    except:
        return "10s"

def add_metadata_to_mov(filepath, keywords):
    keyword_str = ", ".join(keywords)
    temp_path = filepath.replace(".mov", "_temp.mov")
    command = [
        "ffmpeg", "-i", filepath,
        "-metadata", f"com.apple.quicktime.keyword={keyword_str}",
        "-c", "copy", temp_path, "-y"
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.replace(temp_path, filepath)
    print(f"✅ Metadata added to {os.path.basename(filepath)}: {keyword_str}")

def add_finder_tags(filepath, tags):
    for tag in tags:
        try:
            subprocess.run(["tag", "--add", tag, filepath], check=True)
        except subprocess.CalledProcessError:
            print(f"⚠️ Could not add Finder tag '{tag}' to {filepath}")

def prepare_fcpxml_clip(video_path, keywords):
    basename = os.path.basename(video_path)
    file_path_url = f"file://{os.path.abspath(video_path)}"
    video_id = f"id_{uuid.uuid4().hex[:8]}"
    duration = get_video_duration(video_path)

    fcpxml_assets.append({
        "id": video_id,
        "name": basename,
        "src": file_path_url
    })
    fcpxml_clips.append({
        "name": basename,
        "ref": video_id,
        "keywords": keywords,
        "duration": duration
    })

def export_combined_fcpxml(output_path="combined.fcpxml", event_name="Keyworded Clips"):
    fcpxml = ET.Element("fcpxml", version="1.10")
    resources = ET.SubElement(fcpxml, "resources")

    ET.SubElement(resources, "format", id="r1", name="FFVideoFormat1080p25",
                  frameDuration="100/2500s", width="1920", height="1080", colorSpace="1-1-1 (Rec. 709)")

    for asset in fcpxml_assets:
        asset_elem = ET.SubElement(resources, "asset", id=asset["id"], name=asset["name"],
                                   start="0s", hasAudio="1", hasVideo="1", format="r1")
        ET.SubElement(asset_elem, "media-rep", kind="original-media", src=asset["src"])

    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", name=event_name)

    for clip in fcpxml_clips:
        clip_elem = ET.SubElement(event, "asset-clip", name=clip["name"],
                                  ref=clip["ref"], duration=clip["duration"],
                                  start="0s", offset="0s")
        for word in clip["keywords"]:
            ET.SubElement(clip_elem, "keyword", start="0s", duration=clip["duration"], value=word)

    ET.ElementTree(fcpxml).write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"📁 Exported valid combined FCPXML: {output_path}")

def rename_video(original_path, keywords, frames_analyzed, frame_dir):
    dirname, basename = os.path.split(original_path)
    name, ext = os.path.splitext(basename)
    unique_keywords = list(dict.fromkeys(keywords))[:KEYWORDS_PER_VIDEO]
    keyword_str = "_".join(re.sub(r"[^\w\-]", "", kw) for kw in unique_keywords)
    new_name = f"{name}_{keyword_str}{ext}"
    new_path = os.path.join(dirname, new_name)

    if new_path != original_path:
        os.rename(original_path, new_path)
        print(f"Renamed: {basename} → {os.path.basename(new_path)}")

    create_contact_sheet(frame_dir, os.path.splitext(new_path)[0] + "_sheet.jpg")
    add_metadata_to_mov(new_path, unique_keywords)
    add_finder_tags(new_path, unique_keywords)
    prepare_fcpxml_clip(new_path, unique_keywords)

    json_path = os.path.splitext(new_path)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump({
            "original_filename": basename,
            "new_filename": os.path.basename(new_path),
            "keywords": unique_keywords,
            "frames_analyzed": frames_analyzed
        }, f, indent=2)

    csv_rows.append({
        "original_filename": basename,
        "new_filename": os.path.basename(new_path),
        "keywords": ", ".join(unique_keywords)
    })

    return new_path

def process_video(original_path, keywords, frames_analyzed, frame_dir):
    dirname, basename = os.path.split(original_path)
    name, ext = os.path.splitext(basename)
    unique_keywords = list(dict.fromkeys(keywords))[:KEYWORDS_PER_VIDEO]

    # Use original path as-is
    output_prefix = os.path.join(dirname, name)

    # Save contact sheet
    # create_contact_sheet(frame_dir, f"{output_prefix}_sheet.jpg")

    # Add metadata to the original .mov
    add_metadata_to_mov(original_path, unique_keywords)
    add_finder_tags(original_path, unique_keywords)
    prepare_fcpxml_clip(original_path, unique_keywords)

def main():
    for filename in tqdm(os.listdir(VIDEO_DIR)):
        if not filename.endswith(".mov"):
            continue

        video_path = os.path.join(VIDEO_DIR, filename)
        frame_dir = os.path.join(FRAMES_DIR, os.path.splitext(filename)[0])
        extract_frames(video_path, frame_dir)
        keywords = classify_images(frame_dir)
        keyword_counts = Counter(keywords)
        top_keywords = [k for k, _ in keyword_counts.most_common(KEYWORDS_PER_VIDEO)]

#        rename_video(video_path, top_keywords, FRAMES_PER_VIDEO, frame_dir)
        process_video(video_path, top_keywords, FRAMES_PER_VIDEO, frame_dir)

        for f in os.listdir(frame_dir):
            os.remove(os.path.join(frame_dir, f))
        os.rmdir(frame_dir)

    export_combined_fcpxml(os.path.join(VIDEO_DIR, "combined.fcpxml"))

if __name__ == "__main__":
    main()