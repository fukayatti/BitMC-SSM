# ⚡ TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
**〜 Google Research 2025: シャノン理論限界に迫る不偏オンラインベクトル量子化の Bit-MC-SSM 統合 〜**

---

## 📖 論文概要 (Reference)

* **論文:** [arXiv:2504.19874v1 (2025)](https://arxiv.org/html/2504.19874v1)
* **著者:** Amir Zandieh, Majid Daliri, Majid H., Vahab Mirrokni (Google Research & New York University)
* **タイトル:** *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*

---

## 🔬 1. 数理的背景と革新性

従来のベクトル量子化（Product Quantization / PQ など）はデータに対する事前 k-means クラスタリング（オフライン学習）を必要としていましたが、TurboQuant は **事前学習不要（Data-oblivious）かつ完全オンライン** で動作し、情報理論上の **シャノン下限（Shannon Lower Bound: SLB）の約 2.7倍以内（1-bit では 1.45倍）** という極限の歪み率を達成します。

```
[入力ベクトル x] ───> [ランダム直交回転 Π] ───> [各次元が N(0, 1/d) に集中]
                                                          │
   ┌──────────────────────────────────────────────────────┴──────────────────────┐
   ▼                                                                             ▼
【TurboQuant-MSE】                                                       【TurboQuant-Prod】
 1次元 Lloyd-Max スカラー量子化                                           (b-1)-bit MSE 量子化
 (1-bit / 2-bit / 3-bit / 4-bit)                                                 │
                                                                                 ▼ 残差 r = x - x_mse
                                                                          1-bit QJL (符号射影)
                                                                          => 内積の期待値バイアス 0 (不偏)
```

---

## 📊 2. 数値検証結果 (`tests/test_turboquant.py`)

### ① MSE 歪み率と理論限界の比較 ($d=128$, $N=5,000$)

| ビット幅 $b$ | 理論限界値 (Panter-Dite) | 実測 MSE ($D_{\text{mse}}$) | 歪み減少率 |
| :---: | :---: | :---: | :---: |
| **1-bit** | $\approx 0.360$ | **0.3614** | $1.0\times$ (基準) |
| **2-bit** | $\approx 0.117$ | **0.1164** | **3.1倍 精度向上** |
| **3-bit** | $\approx 0.030$ | **0.0341** | **10.6倍 精度向上** |
| **4-bit** | $\approx 0.009$ | **0.0093** | **38.8倍 精度向上** |

### ② 内積推定の不偏性（Bias = 0 検定）

* **従来の 2-bit MSE 量子化 (Biased Baseline):** 平均バイアス $+0.00076$（内積の大きさに応じて系統的偏りが発生）
* **TurboQuant-Prod 3-bit (1-bit QJL 残差結合):** 平均バイアス **$-0.00032 \approx 0.0000$（完全な不偏推定量）**

### ③ Memory-Caching (MC-SSC) 状態想起テスト ($d=64$, 100 Memories)

* **セマンティック・クラスター復元率:** **100.0%**
* **元ベクトル内積との相関係数 (Pearson Correlation):** **99.67%**

---

## 🛠️ 3. Bit-MC-SSM への導入効果

1. **過去隠れ状態キャッシュの極小化:**
   * 長大なコンテキストを処理する際、32トークンごとのチェックポイント状態 $h \in \mathbb{R}^{d_{\text{state}}}$ を **2.5-bit / 3-bit に圧縮して保持**（メモリ消費量を 1/5 以下に削減）。
2. **高速・不偏な記憶想起:**
   * Top-$k$ 類似度検索において、量子化によるバイアスなしで正確な過去文脈を呼び出し可能。
3. **C++ Zero-GEMM への親和性:**
   * QJL の 1-bit 符号部分は `POPCNT` や `_mm256_xor_si256` などのビット並列命令で超高速演算が可能。
