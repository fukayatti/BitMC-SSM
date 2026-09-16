import json
import os

def create_bit_segnet_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# ✂️ Bit-SegNet: 1.58-bit Bi-Delta-SSM Background Removal Training\n",
                    "このノートブックでは、超省メモリな 1.58-bit 双方向 SSM（Spatial 2D Scan）を用いた画像セグメンテーション（背景削除）モデルの学習を行います。\n",
                    "\n",
                    "GaLore オプティマイザと AMP（自動混合精度）を利用することで、無料枠の Colab T4 GPU や Kaggle Notebook でもフルスクラッチ学習が可能です。"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. 依存ライブラリのインストールとリポジトリの準備"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!pip install -q torch torchvision transformers tqdm pillow\n",
                    "![ -d 'BitMC-SSM' ] || git clone https://github.com/fukayatti/BitMC-SSM.git\n",
                    "%cd BitMC-SSM\n",
                    "!git pull origin main"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. データセットの準備 (Oxford-IIIT Pet)\n",
                    "背景削除のベンチマークとしてよく使われる Oxford-IIIT Pet データセットをダウンロードし、前景/背景の二値マスク画像に変換します。\n",
                    "（合計約1GBのダウンロードがあるため数分かかります）"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!python python/prepare_dataset.py"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. 🚀 Bit-SegNet の学習開始\n",
                    "Oxford-IIIT Pet データセットを用いて学習を開始します。VRAM が厳しい場合は `--use_galore` オプションを有効にしてください。\n",
                    "※ここではデモのため10エポックで回していますが、精度を上げる場合は 50~100エポックに設定してください。"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!python python/train_segnet.py \\\n",
                    "    --data_dir ./data/seg \\\n",
                    "    --output_dir ./checkpoints_seg \\\n",
                    "    --batch_size 16 \\\n",
                    "    --epochs 10 \\\n",
                    "    --lr 5e-4 \\\n",
                    "    --use_galore"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. 🔮 推論のテスト (背景透過画像の生成)\n",
                    "学習したチェックポイントを使って、実際に画像の背景を切り抜いてみましょう！"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "from IPython.display import Image, display\n",
                    "\n",
                    "# データセット内の1枚の画像をテストとして使用します\n",
                    "test_image = \"data/seg/images/Abyssinian_1.jpg\"\n",
                    "output_image = \"result.png\"\n",
                    "\n",
                    "# 最新のチェックポイントを自動取得（手動で指定してもOK）\n",
                    "if not os.path.exists('./checkpoints_seg'):\n",
                    "    print(\"チェックポイントが見つかりません。先に学習を実行してください。\")\n",
                    "else:\n",
                    "    checkpoints = sorted([f for f in os.listdir('./checkpoints_seg') if f.endswith('.pt')])\n",
                    "    if len(checkpoints) == 0:\n",
                    "        print(\"チェックポイントが見つかりません。先に学習を実行してください。\")\n",
                    "    else:\n",
                    "        latest_ckpt = os.path.join('./checkpoints_seg', checkpoints[-1])\n",
                    "        print(f\"推論に使用するチェックポイント: {latest_ckpt}\")\n",
                    "        \n",
                    "        # 推論スクリプトの実行\n",
                    "        !python python/infer_segnet.py \\\n",
                    "            --image {test_image} \\\n",
                    "            --checkpoint {latest_ckpt} \\\n",
                    "            --output {output_image}\n",
                    "    \n",
                    "        # 結果の表示\n",
                    "        print(\"\\n--- 🔮 切り抜き結果 ---\")\n",
                    "        display(Image(filename=output_image))"
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
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    out_path = "docs/train_bit_segnet.ipynb"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"✅ Updated Notebook: {out_path}")

if __name__ == "__main__":
    create_bit_segnet_notebook()
