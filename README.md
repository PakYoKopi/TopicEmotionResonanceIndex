# Paper: Measuring Topic Emotion Resonance Through Precision-First Alignment
## Journal of Social Computing (under review)

> **Topic Emotion Resonance Index** — A composite index for measuring the emotional resonance of YouTube comments on political video content topics.

---

## Table of Contents

- [Overview](#overview)
- [TERI Concept & Formula](#teri-concept--formula)
- [Repository Structure](#repository-structure)
- [Pipeline Flow](#pipeline-flow)
- [NLP Models Used](#nlp-models-used)
- [Configuration](#configuration)
- [Output & Data Schema](#output--data-schema)
- [Dataset Statistics](#dataset-statistics)
- [Environment & Dependencies](#environment--dependencies)

---

## Overview

TERI (*Topic Emotion Resonance Index*) is an NLP-based analytical system for measuring the extent to which comments on political YouTube videos resonate emotionally with the topics discussed in the videos. This project targets two domains:

- **`indo`** — Indonesian political videos (comments in Indonesian & mixed languages)
- **`global`** — Global political videos (English & multilingual comments)

This pipeline includes video selection, transcript quality checks, comment-transcript alignment, sentiment/emotion/toxicity inference, detection of coordinated inauthentic behavior (CIB), and calculation of the final TERI index. Here, CIB refers to heuristic indicators of coordinated inauthentic behavior such as duplication bursts, abnormal author concentration, and cross-video reuse.


---

## TERI Concept & Formula

TERI is constructed from three main components, each of which is normalized using a **robust z-score** within the domain scope:

| Component | Weight | Description |
|---|---|---|
| `tox_comp` | 0.40 | Average comment toxicity level (mean + p90, robust-z normalization) |
| `conflict_comp` | 0.35 | Conflict proxy: average conflict value of comments (mean + p90) |
| `coupling_comp` | 0.25 | Semantic coupling of comments with transcript segments (cosine similarity) |

```
TERI_core = 0.40 × tox_comp + 0.35 × conflict_comp + 0.25 × coupling_comp

The weighted formulation above represents the operational implementation
of the TERI interaction concept described in the paper, where affect and
friction signals jointly contribute to resonance.
```

TERI_total is TERI_core adjusted with the Opinion Resonance (OR) signal.

Opinion Resonance (OR) captures clusters of strongly opinionated comments
that amplify emotional alignment within a discussion thread.

In practice, OR is estimated from the upper-tail distribution of polarity
and emotional intensity among grounded comments..

### Sub-indexes

| Column | Description |
|---|---|
| `TERI_core` | Main index before OR correction |
| `TERI_total` | Main index + OR correction |
| `TERI_debot` | Index with de-bot weight (CIB correction) |
| `TERI_tox_sub` | Contribution of toxicity component |
| `TERI_conflict_sub` | Contribution of conflict component |
| `TERI_coupling_sub` | Contribution of coupling component |
---

## Repository Structure

```
TopicEmotionResonanceIndex/
│
├── main_configuration/             # Main configuration (precision target alignment = 0.85)
│   ├── config_snapshot.json        # Complete snapshot of all run parameters
│   ├── env_snapshot.json           # Python, PyTorch, CUDA versions
│   ├── model_meta.json             # Metadata & label map of all HuggingFace models
│   ├── model_label_maps.json       # Label map to sent/tox/emo categories
│   ├── alignment_threshold_chosen.json  # Selected alignment threshold (calibration)
│   ├── video_status.csv            # Status of each video at each stage of the pipeline
│   ├── video_level_indices.csv     # TERI index per video (main output)
│   └── comment_level_manifest.csv  # Score per comment (alignment + sentiment + emotion + toxicity)
│
├── stricter_configuration/         # Stricter configuration (precision alignment = 1.0)
│   ├── config_snapshot.json
│   ├── env_snapshot.json
│   ├── model_meta.json
│   ├── model_label_maps.json
│   ├── alignment_threshold_chosen.json
│   ├── video_status.csv
│   ├── video_level_indices.csv
│   └── comment_level_manifest.csv
│
└── README.md
```
---

## Pipeline Flow

```
Video Candidates
      │
      ▼
1. Candidate discovery pool before eligibility filtering.
   300 videos (200 Indonesian Domain, 100 Global Domain)
      │
      ▼
2. TOPIC SCREENING
   Embedding transcript chunks vs. other videos to filter
   off-topic videos (precision target: 0.90)
      │
      ▼
3. TRANSCRIPT QC
   Filter low-quality transcripts:
   - Min tokens: 80, Min segments: 5
   - Min alpha ratio: 0.45
   - Max repeated token ratio: 0.35
      │
      ▼
4. CHUNK QC
   Segment transcripts into chunks of ~180 tokens
   (range: 120–220 tokens)
      │
      ▼
5. COMMENT-TRANSCRIPT ALIGNMENT
   Model: paraphrase-multilingual-MiniLM-L12-v2
   Calibrate cosine similarity threshold (precision ≥ 0.85)
   Top-K=2 closest chunks per comment
      │
      ▼
6. AFFECT SCORING (NLP Inference)
   - Sentiment: positive / neutral / negative
   - Emotion: 5–28 classes per model
   - Toxicity: continuous score (single/multi-label)
   Routing model based on comment language
      │
      ▼
7. CIB DETECTION
   Detection of duplicate comments, bursts of activity,
   author concentration, and cross-video reuse
      │
      ▼
8. TERI COMPUTATION
   Robust-z normalization per domain
   Mean + p90 aggregation → TERI_core → TERI_total → TERI_debot
```
---

## NLP Models Used

| Role | Model | Language |
|---|---|---|
| **Similarity (Alignment)** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Multilingual |
| **Sentiment (Indonesian)** | `w11wo/indonesian-roberta-base-sentiment-classifier` | Indonesian |
| **Sentiment (English)** | `cardiffnlp/twitter-roberta-base-sentiment-latest` | English |
| **Sentiment (Multilingual)** | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | Multilingual |
| **Emotion (Indonesian)** | `StevenLimcorn/indonesian-roberta-base-emotion-classifier` | Indonesian (5 classes) |
| **Emotion (English)** | `j-hartmann/emotion-english-distilroberta-base` | English (7 classes) |
| **Emotion (Multilingual)** | `AnasAlokla/multilingual_go_emotions_V1.1` | Multilingual (28 classes) |
| **Toxicity (English)** | `unitary/unbiased-toxic-roberta` | English (16 labels) |
| **Toxicity (Multilingual)** | `unitary/multilingual-toxic-xlm-roberta` | Multilingual |

Model routing is done per comment based on language detection (`lang_bucket`): `id` → Indonesian model, `en` → English model, `other` → multilingual model.

---

## Configuration

This repository includes two sets of run results with different configurations:

### `main_configuration`
- **Alignment precision target**: 0.85
- **Calibration threshold**: `sim_thr = 0.608`, `mar_thr = 0.017`
- **Actual precision**: 0.857, **Recall**: 0.053, **Coverage**: 1.75%
- **Number of videos passed**: 51 (49 Indonesian, 2 global)
- **Number of aligned comments: 12,719

### `stricter_configuration`
- **Alignment precision target: 0.85 (but with a stricter threshold)
- **Calibration threshold: `sim_thr = 0.613`, `mar_thr = 0.117`
- **Actual precision**: 1.0, **Recall**: 0.016, **Coverage**: 0.5%
- **Number of videos passed**: 33 (all Indonesian)
- **Number of aligned comments**: 3,056

> **Note**: A stricter configuration results in significantly lower coverage (0.5% vs. 1.75%) but perfect alignment precision (1.0).

### Main Parameters

| Parameter | Value | Description |
|---|---|---|
| `SEED` | 42 | Global random seed |
| `MAX_LEN` | 512 | Maximum length of model input tokens |
| `BATCH_SIZE` | 32 | Inference batch size |
| `CAP_PER_VIDEO` | 5000 | Maximum comments per video |
| `MIN_PER_VIDEO` | 50 | Minimum comments per video |
| `SAMPLE_MODE` | time_stratified | Comment sampling mode |
| `TERI.NORM_METHOD` | robust_z | Index normalization method |
| `TERI.AGG_MODE` | mean_p90 | Aggregation mode (mean + 90th percentile) |
| `TERI.TOP_FRAC` | 0.1 | Top video fraction for ranking |

---

## Output & Data Schema

### `video_level_indices.csv`

The main file containing TERI scores per video.

| Column | Type | Description |
|---|---|---|
| `domain` | str | `indo` or `global` |
| `videoId` | str | YouTube video ID |
| `n` | int | Number of comments processed |
| `tox_mean` | float | Average toxicity score |
| `tox_p90` | float | 90th percentile of toxicity score |
| `tox_top10` | float | Average toxicity of top 10% |
| `coupling_mean` | float | Average coupling score (cosine similarity) |
| `coupling_p90` | float | 90th percentile of coupling score |
| `valence_mean` | float | Average sentiment valence |
| `polarity_mean` | float | Average sentiment polarity |
| `conflict_mean` | float | Average conflict proxy |
| `conflict_p90` | float | 90th percentile conflict proxy |
| `neg_rate` | float | Proportion of negative comments |
| `pos_rate` | float | Proportion of positive comments |
| `polarization` | float | Sentiment polarization level |
| `TERI_core` | float | Main TERI index |
| `TERI_total` | float | TERI + Opinion Resonance correction |
| `TERI_debot` | float | CIB/bot-corrected TERI |
| `TERI_tox_sub` | float | Sub-score for toxicity contribution |
| `TERI_conflict_sub` | float | Sub-score for conflict contribution |
| `TERI_coupling_sub` | float | Sub-score for coupling contribution |
| `or_share` | float | Proportion of Opinion Resonance comments |
| `or_score_p90` | float | OR score percentile 90 |
| `cib_risk` | float | Coordinated Inauthentic Behavior risk |
| `flags` | str | Flag label (e.g., `burst`) |
| `suspected_frac` | float | Fraction of comments suspected to be bots |
| `rank_core` / `rank_total` / `rank_debot` / `rank_cib` | float | Video rank per metric |

### `comment_level_manifest.csv`

File of scores per comment that has been aligned and evaluated.

| Column | Type | Description |
|---|---|---|
| `domain` | str | Video domain |
| `videoId` | str | Video ID |
| `commentId` | str | YouTube comment ID |
| `lang_bucket` | str | Language bucket: `id`, `en`, `other` |
| `lang_conf` | float | Language detection confidence |
| `sim_top1_align` | float | Cosine similarity to closest chunk |
| `sim_margin_align` | float | Margin between top-1 and top-2 similarity |
| `aligned_keep` | bool | Whether the comment passes the alignment threshold |
| `sent_model_id` | str | Sentiment model used |
| `emo_model_id` | str | Emotion model used |
| `tox_model_id` | str | Toxicity model used |
| `sent_top_label` | str | Dominant sentiment label |
| `sent_top_score` | float | Dominant sentiment score |
| `tox_score` | float | Aggregate toxicity score |
| `tox_top_label` | str | Dominant toxicity label |
| `emo_top_label` | str | Dominant emotion label |
| `emo_top_score` | float | Dominant emotion score |
| `emo_entropy` | float | Emotion distribution entropy |
| `valence` | float | Comment sentiment valence |
| `polarity` | float | Comment sentiment polarity |
| `coupling_sim` | float | Cosine similarity coupling score |
| `coupling_margin` | float | Coupling margin |
| `conflict_proxy` | float | Comment conflict proxy |
| `sarcasm_like` | int | Sarcasm comment flag |
| `weight_sarcasm` | float | Weight after sarcasm correction |

### `video_status.csv`

Track the status of each video at every stage of the pipeline.

| Column | Description |
|---|---|
| `domain` | Video domain |
| `videoId` | YouTube video ID |
| `stage` | Pipeline stage: `topic_screened`, `transcript_qc`, `chunk_qc`, `alignment_screened`, `affect_scored` |
| `status` | `ok` (passed) or `skip` (skipped/discarded) |
| `reason` | Reason for skip (e.g., `topic_off_by_threshold`) |
| `n_ok_chunks` | Number of chunks that passed QC |

---

## Dataset Statistics

### Main Configuration (`main_configuration`)

| Metric | Value |
|---|---|
| Total videos processed | 300 |
| Videos passed pipeline | 51 (17%) |
| Videos in `indo` domain | 49 |
| Videos in `global` domain | 2 |
| Total aligned comments | 12,719 |
| Indonesian language comments (`id`) | 9,063 (71.3%) |
| English language comments (`en`) | 965 (7.6%) |
| Other language comments (`other`) | 2,691 (21.2%) |
| TERI_total average | −0.140 |
| TERI_total min / max | −2.433 / 2.041 |

### Stricter Configuration (`stricter_configuration`)

| Metric | Value |
|---|---|
| Total videos processed | 300 |
| Videos that passed the pipeline | 33 (11%) |
| Videos in the `indo` domain | 33 |
| Videos in the `global` domain | 0 |
| Total aligned comments | 3,056 |
| Indonesian language comments (`id`) | 2,123 (69.5%) |
| Other language comments (`other`) | 854 (28.0%) |
| English language comments (`en`) | 79 (2.6%) |

---

## Environment & Dependencies

Based on `env_snapshot.json`:

| Component | Version |
|---|---|
| Python | 3.12.12 |
| PyTorch | 2.9.0+cu128 |
| Transformers (HuggingFace) | 5.0.0 |
| CUDA | 12.8 |
| Platform | Linux x86_64 (Google Colab / Drive) |

The pipeline is run in a **Google Colab** environment with GPU (CUDA) access, using Google Drive as storage for the dataset, model cache, and output.
<!-- 
## Citation
If you use TERI in your research, please cite:
Yulianto, S. P. R., et al. (2026).
Topic Emotion Resonance Index (TERI) for Indonesian YouTube Discourse.

### License: CC BY-NC 4.0
### Creative Commons Attribution-NonCommercial 4.0
