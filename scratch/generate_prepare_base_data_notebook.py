import json
import os

def create_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🌏 Bit-MC-SSM: 多言語ベースモデル 事前準備ノートブック (Colab / 無料枠)\n",
                "\n",
                "**目的**: 日本語(Wikipedia + くだけたWeb文体)＋英語のバイリンガル・トークナイザと、事前学習用トークン列(`data_bin`)をここ(無料Colab)で作り、RunPod等の課金GPU環境には**300M本体モデルの学習だけ**を持ち込む。\n",
                "\n",
                "この段階(トークナイザ訓練・データ前処理)はCPU/ネットワーク律速でGPUを必要としないため、課金環境の時間を消費しないようにColab側で完結させる。\n",
                "\n",
                "### 手順\n",
                "1. リポジトリ取得 & 依存インストール\n",
                "2. Google Drive マウント (成果物を永続化し、後でRunPodへ転送するため)\n",
                "3. 多言語(日本語Wikipedia + CC-100口語Web + 英語SmolLM)バイトレベルBPEトークナイザ訓練\n",
                "4. 圧縮率チェック(語彙サイズが日本語に対して妥当か検証)\n",
                "5. `ja_en_mix` データセットで事前学習用バイナリを生成"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. リポジトリ取得 & 依存ライブラリ"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!git clone https://github.com/fukayatti/BitMC-SSM.git\n",
                "%cd BitMC-SSM\n",
                "!pip install -q datasets tokenizers tqdm"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Google Drive マウント\n",
                "トークナイザと`data_bin`をDriveに保存しておき、RunPodへはこのフォルダごと転送する。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from google.colab import drive\n",
                "drive.mount('/content/drive')\n",
                "\n",
                "OUT_ROOT = '/content/drive/MyDrive/bitmc_ssm'\n",
                "TOKENIZER_DIR = f'{OUT_ROOT}/tokenizer'\n",
                "DATA_BIN = f'{OUT_ROOT}/data/train_tokens.bin'\n",
                "\n",
                "import os\n",
                "os.makedirs(TOKENIZER_DIR, exist_ok=True)\n",
                "os.makedirs(f'{OUT_ROOT}/data', exist_ok=True)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. 多言語トークナイザ訓練\n",
                "3系統(日本語Wikipedia=正式文体 / 日本語CC-100=くだけたWeb文体 / 英語SmolLM)から学習することで、\n",
                "キーボード入力(IME予測)に必要な口語日本語も、翻訳に必要な英語も、両方うまく圧縮できる語彙を作る。\n",
                "\n",
                "`vocab_size=49152` は「小さすぎて日本語がバイト単位まで分解される」のと「edgeモデルとして語彙埋め込みが重くなりすぎる」のバランスを取った値。次のセルで実際の圧縮率を検証する。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python python/train_tokenizer.py \\\n",
                "    --vocab_size 49152 \\\n",
                "    --ja_wiki_docs 100000 \\\n",
                "    --ja_web_docs 100000 \\\n",
                "    --en_docs 200000 \\\n",
                "    --out_dir {TOKENIZER_DIR}"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. 圧縮率チェック(日本語1文字あたりトークン数)\n",
                "1.0に近いほど非効率(≒バイト単位分解)。0.4〜0.6程度なら実用的な圧縮ができている目安。\n",
                "悪ければ`vocab_size`や`ja_wiki_docs`/`ja_web_docs`を増やして再訓練する。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from tokenizers import ByteLevelBPETokenizer\n",
                "\n",
                "tok = ByteLevelBPETokenizer.from_file(f'{TOKENIZER_DIR}/vocab.json', f'{TOKENIZER_DIR}/merges.txt')\n",
                "\n",
                "samples = [\n",
                "    \"今日はいい天気ですね、散歩にでも行こうかな。\",\n",
                "    \"えー、まじで？それはウケるｗｗｗ\",\n",
                "    \"明日の会議の資料をまだ作成していないので、急いで準備する必要がある。\",\n",
                "]\n",
                "\n",
                "for s in samples:\n",
                "    n_tokens = len(tok.encode(s).ids)\n",
                "    ratio = n_tokens / len(s)\n",
                "    print(f\"chars={len(s):3d}  tokens={n_tokens:3d}  tokens/char={ratio:.2f}  | {s}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. 事前学習用トークン列の生成 (`ja_en_mix`)\n",
                "英語50% / 日本語Wikipedia25% / 日本語CC-100(口語)25% で重み付きラウンドロビン混合。\n",
                "`--num_samples`は目標トークン数から逆算して調整する(まずは小さめで動作確認→本番は大きく)。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python python/preprocess_data.py \\\n",
                "    --dataset ja_en_mix \\\n",
                "    --tokenizer_dir {TOKENIZER_DIR} \\\n",
                "    --num_samples 500000 \\\n",
                "    --out {DATA_BIN}"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ✅ 完了 — 次のステップ\n",
                "`Google Drive: bitmc_ssm/tokenizer/` と `bitmc_ssm/data/train_tokens.bin` をRunPodへ転送し、\n",
                "`python/train.py --vocab_size 49152 --data_bin train_tokens.bin ...` で300M本体モデルの学習を開始する。"
            ]
        }
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    os.makedirs('docs', exist_ok=True)
    with open('docs/prepare_base_data_colab.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print("✅ Created Notebook: docs/prepare_base_data_colab.ipynb")

if __name__ == "__main__":
    create_notebook()
