import json
import os

def create_bit_segnet_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# ✂️ U-Bit-SegNet: 1.58-bit 最強背景削除モデル学習\n",
                    "このノートブックでは、超省メモリな 1.58-bit双方向SSMを用いた画像セグメンテーションモデルの学習を行います。\n",
                    "高品質な **DIS5Kデータセット** を使用し、Colabのタイムアウト対策として Google Drive を活用した最強の学習ワークフローを提供します。"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. 📂 Google Drive のマウント\n",
                    "学習済みのモデルや、重いデータセットを退避させるために Google Drive をマウントします。"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from google.colab import drive\n",
                    "drive.mount('/content/drive')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. ⚙️ ライブラリのインストールと最新コードの取得"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!pip install -q torch torchvision transformers tqdm pillow datasets\n",
                    "![ -d 'BitMC-SSM' ] || git clone https://github.com/fukayatti/BitMC-SSM.git\n",
                    "%cd BitMC-SSM\n",
                    "!git pull origin main"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. 📦 【初回のみ】DIS5Kデータの解凍と事前リサイズ\n",
                    "**※すでにGoogle Driveに `dis5k_256.zip` を作成済みの場合は、このセルはスキップしてください。**\n",
                    "\n",
                    "Drive上にある未リサイズの巨大なデータ（`dis5k.zip`）をローカルに持ってきて解凍し、あらかじめ `256x256` にリサイズしてから、再度Zip化して Google Drive に退避させます。"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. 巨大な未リサイズのZipをローカルにコピーして解凍\n",
                    "!cp /content/drive/MyDrive/BitMC-SSM/dis5k.zip /content/\n",
                    "!unzip -q -o /content/dis5k.zip -d /\n",
                    "\n",
                    "# 2. CPU負荷対策として全画像を256x256に事前リサイズ\n",
                    "!python python/resize_dataset.py --data_dir \"/content/data/dis5k\" --out_dir \"/content/data/dis5k_256\" --size 256\n",
                    "\n",
                    "# 3. 次回以降一瞬で読み込めるように、リサイズ済みのものをZip化してGoogle Driveへ退避\n",
                    "!zip -q -r /content/drive/MyDrive/BitMC-SSM/dis5k_256.zip /content/data/dis5k_256\n",
                    "print(\"✅ 初回セットアップ完了！Driveに軽量な dis5k_256.zip を保存しました。\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. 🚀 【毎回実行】Driveからデータを一瞬で復元\n",
                    "Colabのセッションがリセットされた際も、Driveに保存したZipを展開するだけで、ネットワークダウンロードやリサイズをスキップして即座に学習環境を復元できます。"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!cp /content/drive/MyDrive/BitMC-SSM/dis5k_256.zip /content/\n",
                    "!unzip -q -o /content/dis5k_256.zip -d /\n",
                    "print(\"✅ データの復元が完了しました。学習を開始できます。\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. 🧠 U-Bit-SegNet の学習開始\n",
                    "すでに途中まで学習したモデルが Drive にある場合は、`--resume` に自動でパスが渡されて途中から再開します。\n",
                    "初めて学習する場合は `--resume` にファイルが見つからなくても自動で最初からスタートします。"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!python python/train_segnet.py \\\n",
                    "    --data_dir \"/content/data/dis5k_256\" \\\n",
                    "    --output_dir \"/content/drive/MyDrive/BitMC-SSM/checkpoints_dis5k\" \\\n",
                    "    --resume \"/content/drive/MyDrive/BitMC-SSM/checkpoints_dis5k/bit_segnet_best.pt\" \\\n",
                    "    --img_size 256 \\\n",
                    "    --batch_size 64 \\\n",
                    "    --epochs 50 \\\n",
                    "    --base_dim 32 \\\n",
                    "    --depths \"1,1,2,1\""
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. 🔮 推論テスト (画像の背景切り抜き)\n",
                    "学習した最高性能のチェックポイント(`bit_segnet_best.pt`)を使って、画像を切り抜きます。"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from IPython.display import Image, display\n",
                    "\n",
                    "!python python/infer_segnet.py \\\n",
                    "    --image \"/content/data/dis5k_256/val/images/1.jpg\" \\\n",
                    "    --checkpoint \"/content/drive/MyDrive/BitMC-SSM/checkpoints_dis5k/bit_segnet_best.pt\" \\\n",
                    "    --output \"result.png\"\n",
                    "\n",
                    "display(Image(\"result.png\"))"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open('bit_segnet.ipynb', 'w') as f:
        json.dump(nb, f, indent=2)
    print("Created bit_segnet.ipynb with DIS5K Google Drive workflow.")

if __name__ == '__main__':
    create_bit_segnet_notebook()
