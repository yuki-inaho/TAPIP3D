# LLMオンボーディングサマリー

> 新任LLMエージェントが TAPIP3D（本フォーク）に参加する際の初期資料。
> 表記凡例: `確認済み`=本環境で実行/検証済み、`未検証`=コードはあるが未実行、`未確認`=情報源が見つからず。

## 1. プロジェクト概要と目的
- **プロジェクト名称・領域:** TAPIP3D — *Tracking Any Point in Persistent 3D Geometry*（NeurIPS 2025）。単眼RGB / RGB-D 動画における長期 feed-forward 3D 点追跡。
- **最終成果物:** 学習済みチェックポイントによる、各フレームの3D点軌跡 (`coords`) と可視性 (`visibs`) の推論。
- **リポジトリ:** `yuki-inaho/TAPIP3D`（upstream: `zbw001/TAPIP3D` の fork）。本書は **`gtx1070-cu124` ブランチ**＝GTX 1070 で uv 管理にて推論を動かす整備を対象とする。
- **ビジネス背景・価値:** 未確認（研究用途。論文 arXiv:2504.14717 / project page tapip3d.github.io）。
- **現時点の進捗サマリ（確認済み）:**
  - uv 化完了（`pyproject.toml` + `uv.lock`、torch 2.4.1+cu124）。
  - **RGB-D（深度既知）.npz 推論 example が完走**（`demo_inputs/dexycb.npz`、72フレーム、factor=1/2 両方）。
  - `pytest` 9 passed（CPU helper + 実GPUスモーク）。
  - 推論時 GPU 消費を実測（§2・付録）。
  - monocular（MegaSAM）パスは方針により不使用。

## 2. クリティカルな要求・制約
> 「壊してはいけない」ライン。

- **GPU 実行必須**（ユーザー指示）。本マシンは **NVIDIA GeForce GTX 1070 (8GB, Pascal, compute capability 6.1, bfloat16 ネイティブ非対応)**。
- **CUDA 拡張 `pointops2` は sm_61 でビルド必須**。未対応バイナリだと推論時に `CUDA error: no kernel image is available for execution on the device`（upstream issue #3）。
- **`uv sync` は `uv.lock` に無いパッケージを削除する** → 手元ビルドの `pointops2` が毎回消える。sync のたびに `scripts/build_extensions.sh` を再実行すること。
- **bfloat16 を直接使わない**。`torch.cuda.is_bf16_supported()` は GTX 1070 でも True を返す（エミュレーション）ため信頼不可。`inference.py` は compute capability で判定し、sm_80 未満では **fp16** を自動選択（`--precision auto`）。
- **推論は RGB-D（深度既知）.npz パスを既定**にする。monocular（MegaSAM）は遅く、外部チェックポイント（`megasam_final.pth` 等、入手先がREADME未記載）が必要なため常用しない。
- **VRAM 8GB に収める**（実測は付録）。`--resolution_factor 2` で reserved 約 6.2GB と上限に近い。

## 3. 参照すべき資料
| 種別 | ファイル/リンク | 概要・用途 |
|------|------------------|------------|
| セットアップ/使い方 | `README.md` | uv インストール、demo、テスト、古いGPU向け注意・VRAM表 |
| データ準備 | `DATASET.md` | 学習/評価データセットの配置 |
| 依存定義 | `pyproject.toml` / `uv.lock` | uv 管理の依存。torch* は cu124 index 固定 |
| 拡張ビルド | `scripts/build_extensions.sh` | pointops2 を任意の `TORCH_CUDA_ARCH_LIST` でビルド |
| テスト資産 | `tests/` (`test_inference_utils.py`, `test_precision.py`, `test_smoke_gpu.py`) + `conftest.py` | CPU helper + GPUスモーク |
| 推論入口 | `inference.py` → `utils/inference_utils.py` → `models/point_tracker_3d.py` | 推論の本流 |
| 既知課題 | upstream issues (`gh issue list --repo zbw001/TAPIP3D`) | #3 (sm_61), #8 (bf16精度) 等 |
| 環境メモ | `docs/ONBOARDING.md`（本書） | LLM向けオンボーディング |

## 4. タスク境界（任せること / 任せないこと）
### 任せるタスク
- RGB-D（.npz）推論の実行・検証、出力 npz の確認。
- `pytest` の追加・実行、CPU helper の修正。
- uv 依存の調整、`pointops2` の sm_61 再ビルド。
- README / 本ドキュメントの更新。

### 任せないタスク
- monocular（MegaSAM）パスの常用・前提化（方針外。必要時は明示合意のうえ best-effort）。
- 学習（`scripts/train.sh`）の本マシン実行（8×L40S/SLURM 想定。GTX 1070 では非現実的）。
- チェックポイント（`*.pth`）やデータ（`*.npz`）のコミット/再配布（`.gitignore` 済み）。
- 無断の `git push` / 外部公開（実施前に確認）。

## 5. インタラクション方針
- **回答スタイル:** 日本語、簡潔。見出し＋箇条書き、必要に応じて表。
- **回答手順:** 前提（制約確認）→ 根拠（コード/実測）→ 提案または実行。
- **禁止事項・注意:** 未確定事項を断定しない。未検証コマンドを「確認済み」と書かない。**GPU 前提・VRAM 8GB・bf16非対応を常に念頭に置く**。
- **秘匿情報の扱い:** トークン/鍵を出力・コミットしない。チェックポイント・データは管理外。

## 6. 試行タスク（オンボーディング演習）
1. **環境再現:** `uv sync --group dev` → `bash scripts/build_extensions.sh` → `uv run pytest`。**9 passed** になることを確認（`確認済み` の期待値）。
2. **RGB-D 推論:** `uv run python inference.py --input_path demo_inputs/dexycb.npz --checkpoint checkpoints/tapip3d_final.pth --resolution_factor 1` を完走させ、`outputs/inference/.../dexycb.result.npz` の `coords (72, N, 3)` を確認。
3. **精度切替の観察:** `--precision fp16|fp32` を変えて、ログの自動選択とピーク VRAM の差を比較する。

## 7. 運用ルール・変更管理
- **ドキュメント更新:** `確認済み / 未検証 / 未確認` を明記。実測値には計測条件（解像度・フレーム数・dtype）を併記。
- **TBD の扱い:** `未確認` と書き、次に確認すべき情報源を添える。
- **レビュー/承認フロー:** 未確認。GTX 1070 向け変更は `gtx1070-cu124` ブランチに隔離済み。`main` への取り込み可否は要相談。
- **その他:** `uv sync` 実行後は必ず CUDA 拡張を再ビルド（§2）。

---

### 付録: 参考情報

**推論時 GPU 消費（確認済み・GTX 1070, `demo_inputs/dexycb.npz` 72フレーム, 32×32 query grid, fp16）**

| `--resolution_factor` | 推論解像度 | torch allocated | torch reserved | nvidia-smi ピーク |
|---|---|---|---|---|
| 1 | 384×512 | 2.97 GB | 4.10 GB | 5183 MiB |
| 2 | 543×724 | 5.85 GB | 6.20 GB | 7035 MiB |

- baseline（他プロセス）約 1.1GB を含む nvidia-smi 値。いずれも 8GB に収まる（factor=2 は余裕約1GB）。
- 推論時間: 約 3.5 分 / 72 フレーム（fp16, GTX 1070, `確認済み`）。

**主要リポジトリ/ディレクトリ**
- `models/`（`point_tracker_3d.py`, `corr_features/`=pointops2 使用, `encoders/`, `point_updaters/`）
- `utils/`（`inference_utils.py`）, `annotation/`（megasam ほか）, `configs/`, `tests/`, `scripts/`
- `third_party/`（`pointops2`=要ビルド, `cotracker`=純Python, `megasam`=monocular用）

**代表的なコマンド**
```bash
uv sync --group dev                 # コア依存（torch cu124 等）
bash scripts/build_extensions.sh    # pointops2 を sm_61 でビルド（uv sync のたびに）
uv run pytest                       # テスト一式（GPUスモーク含む）
uv run python inference.py --input_path demo_inputs/dexycb.npz \
  --checkpoint checkpoints/tapip3d_final.pth --resolution_factor 1
```

**依存ライブラリ（主）**
- torch 2.4.1+cu124 / torchvision 0.19.1 / torchaudio 2.4.1（pytorch cu124 index）
- requirements 由来: hydra-core, einops, kornia, opencv-python, rerun-sdk~=0.21, transformers==4.57.3, accelerate==1.0.1 ほか
- optional（monocular）: xformers（sm_61 で `memory_efficient_attention` 動作は `確認済み`）

**チェックポイント**
- `checkpoints/tapip3d_final.pth`（HF: `zbww/tapip3d`）— RGB-D 推論に必須。
- monocular 用（`megasam_final.pth` ほか）は未取得・`未検証`。

**連絡先/責任者:** 未確認（git user: `yuki-inaho`）。
