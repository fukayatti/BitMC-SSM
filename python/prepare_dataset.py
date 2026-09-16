import os
import urllib.request
import tarfile
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

def download_and_extract(url, extract_to):
    """URLからファイルをダウンロードして解凍する"""
    filename = url.split('/')[-1]
    filepath = os.path.join(extract_to, filename)
    
    if not os.path.exists(filepath):
        print(f"⬇️ Downloading {filename}...")
        # プログレスバー付きでダウンロード
        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(url, filepath, reporthook=t.update_to)
            
    print(f"📦 Extracting {filename}...")
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=extract_to)
    return filepath

def main():
    print("🚀 Oxford-IIIT Pet データセットをダウンロード・変換します...")
    
    # 作業ディレクトリの準備
    base_dir = Path("data")
    base_dir.mkdir(exist_ok=True)
    
    # 最終的な出力先
    out_img_dir = base_dir / "seg" / "images"
    out_mask_dir = base_dir / "seg" / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    # 1. ダウンロード
    img_url = "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz"
    anno_url = "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz"
    
    download_and_extract(img_url, base_dir)
    download_and_extract(anno_url, base_dir)

    # 2. 変換処理 (Trimap -> Binary Mask)
    raw_img_dir = base_dir / "images"
    raw_trimap_dir = base_dir / "annotations" / "trimaps"
    
    trimap_files = [f for f in os.listdir(raw_trimap_dir) if f.endswith('.png') and not f.startswith('.')]
    
    print("🔄 画像とマスクを背景削除用に変換中 (Trimap -> Binary Mask)...")
    valid_count = 0
    for mask_filename in tqdm(trimap_files):
        stem = Path(mask_filename).stem
        img_filename = f"{stem}.jpg"
        
        raw_img_path = raw_img_dir / img_filename
        raw_mask_path = raw_trimap_dir / mask_filename
        
        if not raw_img_path.exists():
            continue
            
        try:
            # 1 (前景), 2 (背景), 3 (不明/境界) 
            trimap = np.array(Image.open(raw_mask_path))
            
            # 前景(1)だけを255(白)にし、それ以外を0(黒)にするバイナリマスク作成
            binary_mask = (trimap == 1).astype(np.uint8) * 255
            mask_img = Image.fromarray(binary_mask)
            
            # コピー＆保存
            img = Image.open(raw_img_path).convert("RGB")
            
            # リサイズ(256x256)して保存すると学習データ容量を抑えられますが、
            # 今回は train_segnet.py 側でリサイズされるためオリジナルサイズで保存します
            img.save(out_img_dir / img_filename)
            mask_img.save(out_mask_dir / mask_filename)
            valid_count += 1
            
        except Exception as e:
            # 破損ファイルなどはスキップ
            continue
            
    print(f"\n✅ 完了! {valid_count} 枚の画像とマスクペアを以下のディレクトリに準備しました。")
    print(f"📸 画像: {out_img_dir}")
    print(f"🎭 マスク: {out_mask_dir}")
    print("\n👉 そのまま学習を開始できます:")
    print("   python python/train_segnet.py --data_dir data/seg")

if __name__ == "__main__":
    main()
