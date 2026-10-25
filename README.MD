# Replacing the SOFTS Encoder with DINOv3 for Hanoi Weather Forecasting

This repository documents an experimental modification of the SOFTS time-series forecasting model. The experiment replaces the original SOFTS input-embedding/STAR-encoder path with a DINOv3 vision encoder and a new forecasting head. The task is to forecast the next 24 hours for five hourly weather variables from the previous 168 hours.

> **Main result:** The DINOv3-based experiment ran successfully, but its MAE and RMSE were higher than those of the original SOFTS model for all five variables on the held-out test set. The original SOFTS checkpoint is therefore retained as the preferred model for the prototype website. The DINOv3 branch is kept for the architecture comparison and experiment report.

## 1. What Was Changed?

### Original SOFTS

```text
Hourly weather values (168 hours x 5 variables)
        |
        v
DataEmbedding_inverted
        |
        v
SOFTS encoder (EncoderLayer / STAR)
        |
        v
Original projection layer
        |
        v
Forecast: 24 hours x 5 variables
```

### Experimental SOFTS-DINOv3

```text
Hourly weather values (168 hours x 5 variables)
        |
        v
Render each weather window as a 224 x 224 RGB line-chart image
        |
        v
DINOv3 ViT-S/16 encoder (pretrained; frozen during this experiment)
        |
        v
384-dimensional CLS feature vector
        |
        v
New forecasting head:
LayerNorm(384) -> Linear(384, 256) -> GELU -> Dropout(0.1)
-> Linear(256, 24 x 5)
        |
        v
Forecast: 24 hours x 5 variables
```

The experimental model is implemented in **`models/SOFTS_DINOv3.py`**. It does not instantiate or use the STAR module. It also does not use SOFTS's original `DataEmbedding_inverted` path. Because DINOv3 expects images rather than numeric time-series tensors, a conversion step renders the five standardized weather series as vertically stacked line plots before passing the image to DINOv3.

This is a custom research experiment, not an official model released by the SOFTS or DINOv3 authors.

## 2. Model Configuration

- **Original baseline:** SOFTS, with the original STAR-based encoder.
- **Experimental encoder:** `facebook/dinov3-vits16-pretrain-lvd1689m` (DINOv3 ViT-S/16).
- **DINOv3 feature size:** 384.
- **Input window:** 168 hourly steps.
- **Forecast horizon:** 24 hourly steps.
- **Input variables:** temperature, relative humidity, precipitation, wind speed, and cloud cover.
- **Frozen encoder:** DINOv3 parameters were frozen in the reported experiment; only the new forecasting head was trained.
- **Forecasting head:** `LayerNorm(384) -> Linear(384, 256) -> GELU -> Dropout(0.1) -> Linear(256, 120)`, reshaped to `(24, 5)`.
- **Training:** AdamW, learning rate `0.001`, weight decay `0.0001`, batch size `512`, maximum `30` epochs, early-stopping patience `5`, random seed `2026`.
- **Best validation epoch:** 18, with validation loss `0.628986` (MSE on standardized targets).

The DINOv3 model repository is access-controlled. Users must have Hugging Face access and authenticate before downloading the pretrained weights. Do not commit access tokens or the pretrained weights to this repository.

## 3. Dataset

The experiment uses hourly historical weather data downloaded from the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), based on ERA5 reanalysis for a grid point near Hanoi.

- **Rows:** 753,883.
- **Date range in the source CSV:** 1940–2025.
- **Timestamp column:** `date` (UTC).
- **Variables:**

| Column | Description | Unit |
|---|---|---|
| `temperature_2m` | Air temperature at 2 m | °C |
| `relative_humidity_2m` | Relative humidity | % |
| `precipitation` | Precipitation | mm |
| `wind_speed_10m` | Wind speed at 10 m | km/h |
| `cloud_cover` | Cloud cover | % |

ERA5 is reanalysis data, not direct measurements from a specific weather station and not an operational weather forecast. The reported results evaluate time-series predictions on this historical dataset; they do not establish real-world forecast accuracy for dates after the dataset ends.

The CSV is intentionally not included in the repository. To reproduce the experiment, place the file at `dataset/custom/hanoi_weather.csv`. The first column must be `date`, followed by the five variables listed above. Fit `StandardScaler` using only the training portion, then apply the fitted scaler to validation, test, and inference data.

Please follow the data provider's applicable terms and attribution requirements when using or redistributing the dataset.

## 4. Time-Based Splits and Evaluation

The chronological split is 70% training, 10% validation, and 20% test. The data is not randomly shuffled across time. Each sample uses the previous 168 hours to predict the following 24 hours.

| Split | Window stride | Number of windows |
|---|---:|---:|
| Train | 6 hours | 87,921 |
| Validation | 24 hours | 3,140 |
| Test | 24 hours | 6,281 |

Historical context may extend 168 hours before a validation/test boundary so that the model has sufficient input context; the target horizons remain in their respective chronological periods. MAE and RMSE below are computed after converting predictions back to the original units.

## 5. Results: Original SOFTS vs. DINOv3 Encoder

Lower MAE and RMSE indicate better results. Both models were compared over the same test windows.

| Weather variable | SOFTS MAE | DINOv3 MAE | SOFTS RMSE | DINOv3 RMSE | Increase in DINOv3 MAE |
|---|---:|---:|---:|---:|---:|
| Temperature (°C) | 1.3154 | 2.3762 | 1.8124 | 3.0593 | 80.64% |
| Relative humidity (%) | 5.5028 | 8.9375 | 7.4068 | 11.2637 | 62.42% |
| Precipitation (mm) | 0.2441 | 0.2690 | 0.7633 | 0.7817 | 10.19% |
| Wind speed (km/h) | 2.3621 | 2.7105 | 3.1688 | 3.4838 | 14.75% |
| Cloud cover (%) | 18.2652 | 23.6597 | 25.6142 | 28.9499 | 29.53% |

**Conclusion:** In this configuration, the DINOv3-based model had higher MAE and RMSE than the original SOFTS model on all five variables. The smallest MAE gap was for precipitation, and the largest was for temperature. This result does not prove that DINOv3 cannot be useful for any time-series task; it shows that the specific frozen-DINOv3, chart-image representation, and new-head design did not outperform SOFTS on this dataset.

Possible explanations include a mismatch between a vision-pretrained encoder and numerical time-series forecasting, information loss when numeric sequences are rendered as images, and the fact that DINOv3 was frozen rather than fine-tuned. These are plausible explanations, not separately verified causal findings.

## 6. Environment and Reproduction

The experiment was run in Google Colab with an NVIDIA Tesla T4 GPU. Versions observed during the experiment included:

- PyTorch `2.11.0+cu130`
- Transformers `4.57.6`
- Hugging Face Hub `0.36.2`

Additional libraries include NumPy, pandas, scikit-learn, Pillow, and Matplotlib.

A reproduction workflow should:

1. Clone the project and install the required Python packages.
2. Mount Google Drive in Colab and place `hanoi_weather.csv` at the expected path.
3. Authenticate to Hugging Face with an account that has access to the DINOv3 repository.
4. Load the DINOv3 ViT-S/16 pretrained model.
5. Apply chronological splits and fit the scaler only on the training split.
6. Render each 168-hour window as an RGB image and extract/cache the DINOv3 CLS feature for train, validation, and test.
7. Train the forecasting head, select its checkpoint using validation loss, and evaluate on the held-out test split.
8. Compare MAE and RMSE against the saved SOFTS baseline predictions.

For full reproducibility, add the cleaned Google Colab notebook as `notebooks/softs_dinov3_experiment.ipynb`, alongside this model file and the metrics CSV. The notebook should contain data preparation, feature extraction, head training, and evaluation cells in execution order.

## 7. Recommended Repository Layout

```text
models/
  SOFTS.py                         # Original SOFTS implementation
  SOFTS_DINOv3.py                  # Experimental DINOv3 encoder replacement
notebooks/
  softs_dinov3_experiment.ipynb    # Reproducible experiment notebook
results/
  softs_vs_dinov3_metrics.csv      # MAE/RMSE comparison table
README.md                          # This document
LICENSE                            # Retain the upstream SOFTS license and notices
```

Do not commit the raw weather CSV, intermediate feature caches (`*.npz`), model checkpoints, Hugging Face tokens, or other large/generated files unless there is a clear reason and their licensing/access conditions have been reviewed. They can remain in Google Drive for the experiment.

## 8. Which Model Should Be Used for the Website?

Given the current evaluation, use the **original SOFTS checkpoint** for the prototype website because it produced lower errors in this experiment. Keep the DINOv3 branch as a separate architecture experiment that demonstrates what was replaced and how the result differs. Do not describe DINOv3 as more accurate for this dataset.

## 9. References and Attribution

- **SOFTS:** [Official repository](https://github.com/Secilia-Cxy/SOFTS). The upstream repository includes its license and attribution notices; retain them when distributing modified source code.
- **DINOv3:** [Official repository](https://github.com/facebookresearch/dinov3) and [DINOv3 ViT-S/16 on Hugging Face](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m). Review the model license and access conditions before redistribution or deployment.
- **Weather data:** [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), based on ERA5 reanalysis.
