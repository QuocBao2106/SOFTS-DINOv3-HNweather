# HanoiWeather AI — Hướng dẫn chạy và deploy

Bộ này bổ sung backend API và frontend cho repository SOFTS bạn đã clone. `app.py` dùng trực tiếp `models/SOFTS.py` có sẵn, vì vậy cần đặt bộ file này **vào thư mục gốc** của repo `SOFTS-DINOv3-HNweather` (cùng cấp với thư mục `models`, `layers`, `utils`, `exp`).

## 1. Thêm model đã xuất

Lấy đúng 3 file từ Google Drive của bạn:

`SOFTS-HNweather/checkpoints/HanoiWeather_SOFTS_FINAL_168_24/`

- `checkpoint.pth`
- `scaler.pkl`
- `config.json`

Đặt cả ba file tại cùng thư mục tương ứng trong repository local:

`checkpoints/HanoiWeather_SOFTS_FINAL_168_24/`

**Quan trọng:** checkpoint, scaler và config phải thuộc cùng một lần huấn luyện. Không trộn scaler từ `HanoiWeather_SOFTS_168_24` với checkpoint từ `HanoiWeather_SOFTS_FINAL_168_24`.

## 2. Chạy local trên Windows

Mở Terminal tại thư mục gốc repository:

```powershell
# Dùng virtual environment hiện có
pip install -r requirements-web.txt

# Chỉ nếu môi trường chưa cài PyTorch tương thích, cài bản CPU:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Mở website: <http://127.0.0.1:8000>

Tình trạng backend: <http://127.0.0.1:8000/api/health>

Swagger API: <http://127.0.0.1:8000/docs>

Nếu muốn dùng GPU local, cài phiên bản PyTorch CUDA phù hợp với máy của bạn thay vì gói CPU.

## 3. Kiểm tra trước khi deploy

- `/api/health` phải trả `status: ok`.
- `/api/status` phải trả `model_loaded: true`.
- Trang Tổng quan hiển thị thời tiết hiện tại từ Open-Meteo.
- Tab “Dự báo 24h” phải trả kết quả từ `/api/forecast/ai`.
- Nếu `model_loaded: false`, xem thông báo `model_error`, kiểm tra đường dẫn ba artifact, cấu hình và các file mã nguồn SOFTS.

Website vẫn có thể hiển thị current/daily forecast của Open-Meteo khi model artifact chưa có, nhưng dự báo AI 24 giờ cần checkpoint + scaler + config.

## 4. Deploy bằng Render

1. Commit và push `app.py`, `static/index.html`, `requirements-web.txt`, `render.yaml`, `.python-version` cùng ba artifact model nhỏ vào repository GitHub của bạn.
2. Trong Render, tạo **New → Web Service**, kết nối repository `QuocBao2106/SOFTS-DINOv3-HNweather`.
3. Chọn nhánh `main` sau khi các file app/model đã được merge vào đó.
4. Render sẽ đọc `render.yaml`, cài dependencies và chạy Uvicorn. Khi build/deploy thành công, Render cấp URL `https://<service-name>.onrender.com`.
5. Mở `/api/health` và website để kiểm tra. URL deploy chỉ tồn tại sau khi bạn tạo dịch vụ trong tài khoản Render; file này không tự tạo hosting.

Nếu checkpoint lớn hơn 100 MB, không commit trực tiếp file đó lên GitHub thông thường; dùng Git LFS hoặc kho artifact được kiểm soát và cấu hình `MODEL_DIR`/download an toàn.

## 5. Về nguồn dữ liệu và huấn luyện lại

- Điều kiện hiện tại và dự báo nhiều ngày lấy từ Open-Meteo Forecast API.
- Dự báo SOFTS dùng 168 giờ dữ liệu theo giờ mới nhất có thể lấy từ Open-Meteo rồi tạo 24 giờ tiếp theo.
- Mô hình đang triển khai được huấn luyện trên ERA5 đến cuối năm 2025. Đầu vào live từ Open-Meteo không hoàn toàn cùng nguồn phân phối với ERA5, nên cần đánh giá lại sai số thực tế trước khi quảng bá độ chính xác.
- Bản này **không tự huấn luyện lại khi có người truy cập**. Huấn luyện nên chạy như một quy trình riêng có kiểm tra dữ liệu, đánh giá validation/test và chỉ thay checkpoint đang phục vụ sau khi đạt điều kiện.

Nguồn: [Open-Meteo](https://open-meteo.com/) và [SOFTS upstream](https://github.com/Secilia-Cxy/SOFTS). Giữ lại thông tin giấy phép và trích dẫn của repository SOFTS gốc.
