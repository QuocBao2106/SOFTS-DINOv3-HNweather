
import numpy as np
import torch
import torch.nn as nn

from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModel


MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"


def weather_window_to_image(window, size=224):
    """
    Chuyển một cửa sổ thời tiết [seq_len, n_features]
    thành ảnh RGB gồm các biểu đồ thời gian xếp dọc.

    window phải được chuẩn hóa bằng scaler đã fit trên train.
    """
    window = np.asarray(window, dtype=np.float32)

    if window.ndim != 2:
        raise ValueError("window phải có dạng [seq_len, n_features].")

    window = np.nan_to_num(
        window, nan=0.0, posinf=3.0, neginf=-3.0
    )

    seq_len, n_features = window.shape
    image = Image.new(
        "RGB", (size, size), color=(250, 250, 250)
    )
    draw = ImageDraw.Draw(image)

    # Mỗi biến có một ô biểu đồ riêng.
    # Màu sắc chỉ dùng để phân biệt các đường.
    colors = [
        (220, 65, 65),    # temperature
        (35, 145, 80),    # humidity
        (40, 105, 220),   # precipitation
        (150, 75, 180),   # wind
        (80, 90, 105),    # cloud cover
    ]

    left, right = 5, size - 6
    top_margin, bottom_margin = 4, 4
    panel_height = size / n_features

    x_coords = np.linspace(
        left, right, seq_len
    ).astype(int)

    for j in range(n_features):
        top = int(j * panel_height)
        bottom = int((j + 1) * panel_height) - 1

        # Đường phân cách giữa các biến
        if j > 0:
            draw.line(
                (0, top, size - 1, top),
                fill=(215, 219, 225),
                width=1
            )

        y_top = top + top_margin
        y_bottom = bottom - bottom_margin

        # Dữ liệu đã chuẩn hóa: giữ cùng một thang đo
        # giữa các cửa sổ, tránh tự co giãn từng cửa sổ.
        values = np.clip(window[:, j], -3.0, 3.0)
        normalized = (values + 3.0) / 6.0

        y_coords = (
            y_bottom
            - normalized * (y_bottom - y_top)
        ).astype(int)

        points = list(zip(
            x_coords.tolist(),
            y_coords.tolist()
        ))

        if len(points) >= 2:
            draw.line(
                points,
                fill=colors[j % len(colors)],
                width=2
            )

    return image


class Model(nn.Module):
    """
    Mô hình thử nghiệm thay encoder STAR của SOFTS bằng DINOv3.
    Đầu vào: [batch, seq_len, n_features].
    Đầu ra: [batch, pred_len, n_features].
    """

    def __init__(self, configs):
        super().__init__()

        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.n_features = int(
            getattr(configs, "enc_in", 5)
        )

        model_id = getattr(
            configs, "dinov3_model_id", MODEL_ID
        )

        # ENCODER THAY THẾ: DINOv3.
        # Không tạo STAR hoặc DataEmbedding_inverted.
        self.processor = AutoImageProcessor.from_pretrained(
            model_id
        )
        self.encoder = AutoModel.from_pretrained(model_id)

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        hidden_size = self.encoder.config.hidden_size

        # Forecasting head mới ánh xạ đặc trưng DINOv3
        # thành pred_len x n_features.
        self.forecast_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(
                256, self.pred_len * self.n_features
            )
        )

        self.encoder.eval()

    def train(self, mode=True):
        # Cho phép head train, nhưng luôn giữ DINOv3 ở eval.
        super().train(mode)
        self.encoder.eval()
        return self

    def encode(self, x_enc):
        """
        Trích xuất CLS embedding của DINOv3 từ cửa sổ thời tiết.
        Trả về [batch, 384] với ViT-S/16.
        """
        if x_enc.ndim != 3:
            raise ValueError(
                "x_enc phải có dạng [batch, seq_len, n_features]."
            )

        if x_enc.shape[1] != self.seq_len:
            raise ValueError(
                f"Cần {self.seq_len} giờ đầu vào, "
                f"nhận được {x_enc.shape[1]}."
            )

        if x_enc.shape[2] != self.n_features:
            raise ValueError(
                f"Cần {self.n_features} biến, "
                f"nhận được {x_enc.shape[2]}."
            )

        # Vẽ ảnh từ dữ liệu số; DINOv3 được đóng băng,
        # vì vậy không cần gradient qua bước chuyển đổi này.
        windows = x_enc.detach().float().cpu().numpy()

        images = [
            weather_window_to_image(window)
            for window in windows
        ]

        pixel_values = self.processor(
            images=images,
            return_tensors="pt"
        )["pixel_values"]

        device = next(self.encoder.parameters()).device
        pixel_values = pixel_values.to(device)

        with torch.no_grad():
            outputs = self.encoder(
                pixel_values=pixel_values
            )
            embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        mask=None
    ):
        # x_enc phải được chuẩn hóa bằng scaler đã fit trên train.
        embeddings = self.encode(x_enc)

        output = self.forecast_head(embeddings)

        return output.reshape(
            x_enc.shape[0],
            self.pred_len,
            self.n_features
        )
