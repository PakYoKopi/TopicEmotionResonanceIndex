# TopicEmotionResonanceIndex (TERI)

> **Topic Emotion Resonance Index** — Sebuah indeks komposit untuk mengukur resonansi emosional komentar YouTube terhadap topik konten video politik.

---

## Daftar Isi

- [Gambaran Umum](#gambaran-umum)
- [Konsep & Formula TERI](#konsep--formula-teri)
- [Struktur Repositori](#struktur-repositori)
- [Alur Pipeline](#alur-pipeline)
- [Model NLP yang Digunakan](#model-nlp-yang-digunakan)
- [Konfigurasi](#konfigurasi)
- [Output & Skema Data](#output--skema-data)
- [Statistik Dataset](#statistik-dataset)
- [Lingkungan & Dependensi](#lingkungan--dependensi)

---

## Gambaran Umum

TERI (*Topic Emotion Resonance Index*) adalah sistem analitik berbasis NLP untuk mengukur sejauh mana komentar pada video YouTube politik beresonansi secara emosional dengan topik yang dibahas dalam video. Proyek ini menargetkan dua domain:

- **`indo`** — Video politik Indonesia (komentar berbahasa Indonesia & campuran)
- **`global`** — Video politik global (komentar berbahasa Inggris & multibahasa)

Pipeline ini mencakup seleksi video, quality-check transkrip, alignment komentar-transkrip, inferensi sentimen/emosi/toksisitas, deteksi perilaku tidak autentik (*Coordinated Inauthentic Behavior* / CIB), dan kalkulasi indeks akhir TERI.

---

## Konsep & Formula TERI

TERI dibangun dari tiga komponen utama yang masing-masing dinormalisasi menggunakan **robust z-score** dalam lingkup domain:

| Komponen | Bobot | Deskripsi |
|---|---|---|
| `tox_comp` | 0.40 | Rata-rata tingkat toksisitas komentar (mean + p90, normalisasi robust-z) |
| `conflict_comp` | 0.35 | Proksi konflik: rata-rata nilai konflik komentar (mean + p90) |
| `coupling_comp` | 0.25 | Keterikatan semantik komentar dengan segmen transkrip (cosine similarity) |

```
TERI_core = 0.40 × tox_comp + 0.35 × conflict_comp + 0.25 × coupling_comp
```

**`TERI_total`** merupakan `TERI_core` yang disesuaikan dengan sinyal *Opinion Resonance* (OR) untuk merefleksikan kelompok komentar beropini kuat. **`TERI_debot`** adalah versi yang telah dikoreksi terhadap estimasi aktivitas bot/CIB.

### Sub-indeks

| Kolom | Keterangan |
|---|---|
| `TERI_core` | Indeks utama sebelum koreksi OR |
| `TERI_total` | Indeks utama + koreksi OR |
| `TERI_debot` | Indeks dengan bobot de-bot (koreksi CIB) |
| `TERI_tox_sub` | Kontribusi komponen toksisitas |
| `TERI_conflict_sub` | Kontribusi komponen konflik |
| `TERI_coupling_sub` | Kontribusi komponen coupling |

---

## Struktur Repositori

```
TopicEmotionResonanceIndex/
│
├── main_configuration/             # Konfigurasi utama (precision target alignment = 0.85)
│   ├── config_snapshot.json        # Snapshot lengkap semua parameter run
│   ├── env_snapshot.json           # Versi Python, PyTorch, CUDA
│   ├── model_meta.json             # Metadata & label map semua model HuggingFace
│   ├── model_label_maps.json       # Peta label ke kategori sent/tox/emo
│   ├── alignment_threshold_chosen.json  # Threshold alignment yang dipilih (kalibrasi)
│   ├── video_status.csv            # Status setiap video di setiap tahap pipeline
│   ├── video_level_indices.csv     # Indeks TERI per video (output utama)
│   └── comment_level_manifest.csv  # Skor per komentar (alignment + sentimen + emosi + toksisitas)
│
├── stricter_configuration/         # Konfigurasi lebih ketat (precision alignment = 1.0)
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

## Alur Pipeline

```
Kandidat Video
      │
      ▼
1. SELEKSI VIDEO
   Stratifikasi berdasarkan tahun & tier komentar
   Target: 200 video/domain
      │
      ▼
2. TOPIC SCREENING
   Embedding chunk transkrip vs. video lain untuk memfilter
   video off-topic (precision target: 0.90)
      │
      ▼
3. TRANSCRIPT QC
   Filter transkrip berkualitas rendah:
   - Min token: 80, Min segmen: 5
   - Min rasio alfa: 0.45
   - Max rasio token berulang: 0.35
      │
      ▼
4. CHUNK QC
   Segmentasi transkrip menjadi chunk ~180 token
   (range: 120–220 token)
      │
      ▼
5. ALIGNMENT KOMENTAR-TRANSKRIP
   Model: paraphrase-multilingual-MiniLM-L12-v2
   Kalibrasi threshold cosine similarity (precision ≥ 0.85)
   Top-K=2 chunk terdekat per komentar
      │
      ▼
6. AFFECT SCORING (Inferensi NLP)
   - Sentimen: positif / netral / negatif
   - Emosi: 5–28 kelas per model
   - Toksisitas: skor kontinu (single/multi-label)
   Routing model berdasarkan bahasa komentar
      │
      ▼
7. CIB DETECTION
   Deteksi komentar duplikat, burst aktivitas,
   konsentrasi author, dan reuse lintas-video
      │
      ▼
8. TERI COMPUTATION
   Normalisasi robust-z per domain
   Agregasi mean + p90 → TERI_core → TERI_total → TERI_debot
```

---

## Model NLP yang Digunakan

| Role | Model | Bahasa |
|---|---|---|
| **Similarity (Alignment)** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Multibahasa |
| **Sentimen (Indonesia)** | `w11wo/indonesian-roberta-base-sentiment-classifier` | Indonesia |
| **Sentimen (Inggris)** | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Inggris |
| **Sentimen (Multibahasa)** | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | Multibahasa |
| **Emosi (Indonesia)** | `StevenLimcorn/indonesian-roberta-base-emotion-classifier` | Indonesia (5 kelas) |
| **Emosi (Inggris)** | `j-hartmann/emotion-english-distilroberta-base` | Inggris (7 kelas) |
| **Emosi (Multibahasa)** | `AnasAlokla/multilingual_go_emotions_V1.1` | Multibahasa (28 kelas) |
| **Toksisitas (Inggris)** | `unitary/unbiased-toxic-roberta` | Inggris (16 label) |
| **Toksisitas (Multibahasa)** | `unitary/multilingual-toxic-xlm-roberta` | Multibahasa |

Routing model dilakukan per komentar berdasarkan deteksi bahasa (`lang_bucket`): `id` → model Indonesia, `en` → model Inggris, `other` → model multibahasa.

---

## Konfigurasi

Repositori ini menyertakan dua set hasil run dengan konfigurasi berbeda:

### `main_configuration`
- **Alignment precision target**: 0.85
- **Threshold kalibrasi**: `sim_thr = 0.608`, `mar_thr = 0.017`
- **Precision aktual**: 0.857, **Recall**: 0.053, **Coverage**: 1.75%
- **Jumlah video lolos**: 51 (49 indo, 2 global)
- **Jumlah komentar ter-align**: 12.719

### `stricter_configuration`
- **Alignment precision target**: 0.85 (namun threshold lebih ketat)
- **Threshold kalibrasi**: `sim_thr = 0.613`, `mar_thr = 0.117`
- **Precision aktual**: 1.0, **Recall**: 0.016, **Coverage**: 0.5%
- **Jumlah video lolos**: 33 (semua indo)
- **Jumlah komentar ter-align**: 3.056

> **Catatan**: Konfigurasi yang lebih ketat menghasilkan coverage yang jauh lebih rendah (0.5% vs 1.75%) namun presisi alignment sempurna (1.0).

### Parameter Utama

| Parameter | Nilai | Keterangan |
|---|---|---|
| `SEED` | 42 | Random seed global |
| `MAX_LEN` | 512 | Panjang maksimum token input model |
| `BATCH_SIZE` | 32 | Ukuran batch inferensi |
| `CAP_PER_VIDEO` | 5000 | Maksimum komentar per video |
| `MIN_PER_VIDEO` | 50 | Minimum komentar per video |
| `SAMPLE_MODE` | time_stratified | Mode sampling komentar |
| `TERI.NORM_METHOD` | robust_z | Metode normalisasi indeks |
| `TERI.AGG_MODE` | mean_p90 | Mode agregasi (rata-rata + persentil 90) |
| `TERI.TOP_FRAC` | 0.1 | Fraksi top video untuk ranking |

---

## Output & Skema Data

### `video_level_indices.csv`

File utama yang berisi skor TERI per video.

| Kolom | Tipe | Keterangan |
|---|---|---|
| `domain` | str | `indo` atau `global` |
| `videoId` | str | ID video YouTube |
| `n` | int | Jumlah komentar yang diproses |
| `tox_mean` | float | Rata-rata skor toksisitas |
| `tox_p90` | float | Persentil 90 skor toksisitas |
| `tox_top10` | float | Rata-rata toksisitas 10% teratas |
| `coupling_mean` | float | Rata-rata skor coupling (cosine sim) |
| `coupling_p90` | float | Persentil 90 skor coupling |
| `valence_mean` | float | Rata-rata valensi sentimen |
| `polarity_mean` | float | Rata-rata polaritas sentimen |
| `conflict_mean` | float | Rata-rata proksi konflik |
| `conflict_p90` | float | Persentil 90 proksi konflik |
| `neg_rate` | float | Proporsi komentar negatif |
| `pos_rate` | float | Proporsi komentar positif |
| `polarization` | float | Tingkat polarisasi sentimen |
| `TERI_core` | float | Indeks TERI utama |
| `TERI_total` | float | TERI + koreksi Opinion Resonance |
| `TERI_debot` | float | TERI terkoreksi CIB/bot |
| `TERI_tox_sub` | float | Sub-skor kontribusi toksisitas |
| `TERI_conflict_sub` | float | Sub-skor kontribusi konflik |
| `TERI_coupling_sub` | float | Sub-skor kontribusi coupling |
| `or_share` | float | Proporsi komentar Opinion Resonance |
| `or_score_p90` | float | Skor OR persentil 90 |
| `cib_risk` | float | Risiko Coordinated Inauthentic Behavior |
| `flags` | str | Label flag (mis. `burst`) |
| `suspected_frac` | float | Fraksi komentar yang dicurigai bot |
| `rank_core` / `rank_total` / `rank_debot` / `rank_cib` | float | Peringkat video per metrik |

### `comment_level_manifest.csv`

File skor per komentar yang telah di-align dan dinilai.

| Kolom | Tipe | Keterangan |
|---|---|---|
| `domain` | str | Domain video |
| `videoId` | str | ID video |
| `commentId` | str | ID komentar YouTube |
| `lang_bucket` | str | Bucket bahasa: `id`, `en`, `other` |
| `lang_conf` | float | Confidence deteksi bahasa |
| `sim_top1_align` | float | Cosine similarity ke chunk terdekat |
| `sim_margin_align` | float | Margin antara top-1 dan top-2 similarity |
| `aligned_keep` | bool | Apakah komentar lolos threshold alignment |
| `sent_model_id` | str | Model sentimen yang digunakan |
| `emo_model_id` | str | Model emosi yang digunakan |
| `tox_model_id` | str | Model toksisitas yang digunakan |
| `sent_top_label` | str | Label sentimen dominan |
| `sent_top_score` | float | Skor sentimen dominan |
| `tox_score` | float | Skor toksisitas agregat |
| `tox_top_label` | str | Label toksisitas dominan |
| `emo_top_label` | str | Label emosi dominan |
| `emo_top_score` | float | Skor emosi dominan |
| `emo_entropy` | float | Entropi distribusi emosi |
| `valence` | float | Valensi sentimen komentar |
| `polarity` | float | Polaritas sentimen komentar |
| `coupling_sim` | float | Skor coupling cosine similarity |
| `coupling_margin` | float | Margin coupling |
| `conflict_proxy` | float | Proksi konflik komentar |
| `sarcasm_like` | int | Flag komentar bersifat sarkasme |
| `weight_sarcasm` | float | Bobot setelah koreksi sarkasme |

### `video_status.csv`

Melacak status setiap video di setiap tahap pipeline.

| Kolom | Keterangan |
|---|---|
| `domain` | Domain video |
| `videoId` | ID video YouTube |
| `stage` | Tahap pipeline: `topic_screened`, `transcript_qc`, `chunk_qc`, `alignment_screened`, `affect_scored` |
| `status` | `ok` (lolos) atau `skip` (dilewati/dibuang) |
| `reason` | Alasan skip (mis. `topic_off_by_threshold`) |
| `n_ok_chunks` | Jumlah chunk yang lolos QC |

---

## Statistik Dataset

### Konfigurasi Utama (`main_configuration`)

| Metrik | Nilai |
|---|---|
| Total video diproses | 300 |
| Video lolos pipeline | 51 (17%) |
| Video domain `indo` | 49 |
| Video domain `global` | 2 |
| Total komentar ter-align | 12.719 |
| Komentar bahasa Indonesia (`id`) | 9.063 (71.3%) |
| Komentar bahasa Inggris (`en`) | 965 (7.6%) |
| Komentar bahasa lain (`other`) | 2.691 (21.2%) |
| TERI_total rata-rata | −0.140 |
| TERI_total min / maks | −2.433 / 2.041 |

### Konfigurasi Lebih Ketat (`stricter_configuration`)

| Metrik | Nilai |
|---|---|
| Total video diproses | 300 |
| Video lolos pipeline | 33 (11%) |
| Video domain `indo` | 33 |
| Video domain `global` | 0 |
| Total komentar ter-align | 3.056 |
| Komentar bahasa Indonesia (`id`) | 2.123 (69.5%) |
| Komentar bahasa lain (`other`) | 854 (28.0%) |
| Komentar bahasa Inggris (`en`) | 79 (2.6%) |

---

## Lingkungan & Dependensi

Berdasarkan `env_snapshot.json`:

| Komponen | Versi |
|---|---|
| Python | 3.12.12 |
| PyTorch | 2.9.0+cu128 |
| Transformers (HuggingFace) | 5.0.0 |
| CUDA | 12.8 |
| Platform | Linux x86_64 (Google Colab / Drive) |

Pipeline dijalankan di lingkungan **Google Colab** dengan akses GPU (CUDA), menggunakan Google Drive sebagai storage untuk dataset, cache model, dan output.