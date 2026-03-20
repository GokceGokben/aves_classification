import os
import zipfile
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

SOURCE_DIR = 'data/nabirds/images'
TARGET_DIR = 'data/nabirds_192x192/images'
ZIP_NAME = 'data/nabirds_192x192.zip'
TARGET_SIZE = (192, 192)

def process_image(args):
    src_path, tgt_path = str(args[0]), str(args[1])
    try:
        if not os.path.exists(tgt_path):
            os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
            img = Image.open(src_path).convert('RGB')
            # Resize with Lanczos interpolation for best quality
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            img.save(tgt_path, "JPEG", quality=85)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False

def main():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        
    print(f"Collecting files from {SOURCE_DIR}...")
    tasks = []
    for root, _, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, SOURCE_DIR)
                tgt_path = os.path.join(TARGET_DIR, rel_path)
                tasks.append((src_path, tgt_path))
                
    print(f"Found {len(tasks)} images. Resizing to {TARGET_SIZE}...")
    
    # Process images using multiprocessing
    with Pool(max(1, cpu_count() - 1)) as p:
        list(tqdm(p.imap(process_image, tasks), total=len(tasks)))
        
    print(f"\nResizing complete! Now zipping the folder to {ZIP_NAME}...")
    
    # Zip the resized images folder
    with zipfile.ZipFile(ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk('data/nabirds_192x192'):
            for file in files:
                file_path = os.path.join(root, file)
                # Ensure we don't zip the zip file itself if it's placed inside
                if file_path != ZIP_NAME:
                    zipf.write(file_path, os.path.relpath(file_path, 'data/nabirds_192x192'))
                    
    print(f"\nDone! The dataset is ready for Colab/Kaggle at: {ZIP_NAME}")
    print(f"Size: {os.path.getsize(ZIP_NAME) / (1024 * 1024):.2f} MB")

if __name__ == '__main__':
    main()
