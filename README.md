# Satellite Image Cloud Segmentation Pipeline

A modular and scalable pipeline for ingesting, storing, and segmenting clouds from satellite imagery using deep learning models. This project supports ingestion of raw satellite datasets, chunked storage in S3-compatible MinIO buckets, and standalone inference with pre-trained models like UNet and SegFormer.

---

## Project Structure
```
.
├── generalised_ingestion.py       # Ingest raw satellite data into MinIO
├── infer_unet.py                  # Standalone inference with UNet model
├── infer_segformer.py             # Standalone inference with SegFormer model
├── train_unet.py                  # Train UNet model using S3-hosted data
├── train_segformer.py             # Train SegFormer model using S3-hosted data
├── requirements.txt               # List of Python dependencies
├── .env.example                   # Template for environment variables
├── cloud_masks_data.py/json       # Cloud mask metadata
├── dataset.py                     # Dataset construction utilities
├── confusion_metrics_demo.ipynb   # Demo: Confusion matrix evaluation
├── dynamic_data_loader_demo.ipynb # Demo: Dynamic data loader pipeline
├── energy_benchmarks.py           # Energy benchmarking
├── time_benchmarks.py             # Time benchmarking
├── unet.py                        # UNet architecture
```
---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/IhsanAli-mia/Satellite-Cloud-Segmentation-Pipeline.git
cd Satellite-Cloud-Segmentation-Pipeline
```

### 2. Install Dependencies

Ensure you’re using a Python 3.8+ environment.

```bash
pip install -r requirements.txt
```

### 3. Setup Environment

Copy the example environment file and fill in the required values:

```bash
cp .env.example .env
```

Set values such as:

- `MINIO_ENDPOINT`
- `MINIO_ACCESS_KEY`
- `MINIO_SECRET_KEY`
- `S3_BUCKET_NAME `
---

## Data Storage Architecture (MinIO)

This project uses a **MinIO server** to simulate an S3-compatible object store.

### Bucket Structure

```
your-bucket-name/
│
├── stac/        # STAC-compliant JSON metadata for each scene
├── zarr/        # Zarr-formatted chunked satellite imagery
└── masks/       # Ground-truth segmentation masks
```

Zarr format ensures **efficient chunkwise access** and streaming, suitable for large-scale training or inference.

---

## Data Ingestion

Use `generalised_ingestion.py` to upload raw data into MinIO.

### Expected Raw Data Format

```
raw_dataset/
├── Scenes/               # Raw satellite image scenes
├── Masks/                # Corresponding segmentation masks
├── Metadata/             # Additional metadata per image
└── data.json             # Description of dataset characteristics
```

### Run Ingestion

```bash
generalised_ingestion.ipynb
```

This script will parse the raw data, generate STAC metadata, convert to Zarr chunks, and upload all files to the respective MinIO paths.

---

## Training

Use `train_unet.py` or `train_segformer.py` to train models using data fetched directly from the S3/MinIO storage.

```bash
python train_unet.py
python train_segformer.py
```

Both scripts assume properly chunked datasets and masks are available in the configured S3 bucket.

## Inference Pipelines

Both `infer_unet.py` and `infer_segformer.py` are standalone scripts — no need to integrate them with the full pipeline.

### Inference with UNet

```bash
python infer_unet.py
```

### Inference with SegFormer

```bash
python infer_segformer.py
```

Each script handles preprocessing, model loading, and mask generation independently.

---
## Demos and Benchmarks

- `dynamic_data_loader_demo.ipynb`: Demonstrates how the data loader dynamically reads and balances Zarr chunks.
- `confusion_metrics_demo.ipynb`: Demonstrates confusion matrix generation from model predictions.
- `energy_benchmarks.py`: Scripts to evaluate energy usage.
- `time_benchmarks.py`: Scripts to evaluate inference/training runtime.

## 📊 Visualization Dashboard

You can explore the characteristics of the ingested data using the associated visualization dashboard repository:

🔗 [Satellite Data Posteriors Dashboard Prototype](https://github.com/IhsanAli-mia/Satellite_Data_Posteriors_Dashboard_Prototype)

This dashboard provides insights into:

- Dataset distribution
- Metadata trends
---

## 📄 License

MIT License

---