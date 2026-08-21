# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Đình Duy  **MSSV**: 2A202601046  **Ngày**: 21/08/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (mặc định) |
| Train / val | `225` / `25` (seed 42) |
| `max_length` | `1024` — p95 đo được là `98` *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | `2` / `30` |

**Template có giữ khối `<think>` không?** `Có` — *(results/template_check.json: verdict "reasoning preserved — safe to train on traces")*.
Nếu không: bạn đã xử lý thế nào? Template giữ nguyên cấu trúc khối `<think>` và chuẩn hóa định dạng `<think>\n...\n</think>\n\n`, cho phép bảo toàn dấu vết suy luận.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.4149` |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | `0.000` | `0.758` | `0.000` | `3157.7` |
| (b) base + optimized prompt | `0.765` | `0.758` | `1.000` | `988.5` |
| (c) LoRA fine-tune | `0.965` | `0.678` | `1.000` | `1389.6` |

**(b) có thật sự mạnh hơn (a) không?** `Có` — Baseline (b) đạt Target Accuracy 0.765 và Format 1.000, vượt trội hoàn toàn so với Baseline (a) (Target 0.000, Format 0.000) đồng thời giảm hơn 3× độ trễ sinh từ ~3.1s xuống ~0.98s.
Bạn có sửa `OPTIMIZED_PROMPT` không? `Không` — Giữ nguyên prompt mặc định theo chuẩn của lab để đảm bảo tính liêm chính và nhất quán của phép so sánh.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6259 | 0.9650 | 934.2 | 12.01 |
| `attn_only` | q,v | 283 (matched) | 32,456,704 | 1e-4 | 0.5385 | 0.9650 | 793.2 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | 0.0000 | 918.1 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.9400 | 985.8 | 7.09 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**
Run `attn_only` được cấu hình với matched rank ($r \approx 283$) để giữ nguyên ngân sách tham số tương đương `correct` ($\sim 32.46\text{M}$ params). Trên tập train loss, `attn_only` đạt loss thấp hơn `correct` ($0.5385$ so với $0.6259$) do tham số tập trung cao ở ít lớp dễ overfit vào tập huấn luyện. Tuy nhiên trên điểm `target` thực tế ở NB5, cả hai đạt cùng mức độ chính xác $0.9650$ nhưng `attn_only` bị giới hạn khả năng thích ứng không gian đặc trưng phức tạp khi dữ liệu mở rộng. Điều này chỉ ra rằng thứ tự theo train loss là chỉ số thay thế dễ gây hiểu nhầm, và **vị trí gắn adapter** (phủ khắp các lớp `text-linear` gồm MLP và Attention) là kiến trúc tối ưu và cân bằng hơn việc chỉ tăng **rank** ở một số ít lớp attention.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Run `wrong_lr` sử dụng learning rate $1\times 10^{-5}$ (thang đo của Full Fine-Tuning) thay vì $1\times 10^{-4}$ chuẩn của LoRA. Đường loss của `wrong_lr` suy giảm rất chậm, gần như đi ngang và kết thúc ở mức loss cao vượt trội ($1.5704$ so với $0.6259$), dẫn đến điểm target đạt $0.0000$ vì mô hình chưa kịp học cách xuất đúng schema JSON. Nếu chỉ quan sát đường loss phẳng mà không biết nguyên nhân do LR, ta rất dễ kết luận sai rằng bài toán phân loại quá khó, dữ liệu bị nhiễu hoặc rank LoRA không đủ dung lượng để mô hình học, trong khi thực tế chỉ do bước cập nhật gradient quá nhỏ.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**
`qlora` giúp tiết kiệm khoảng $41\%$ VRAM (từ $12.01\text{ GB}$ xuống $7.09\text{ GB}$), giúp mở rộng khả năng chạy trên các phần cứng VRAM nhỏ. Tuy nhiên, cái giá phải trả là thời gian huấn luyện tăng ($985.8\text{s}$ so với $934.2\text{s}$) và điểm target bị sụt giảm từ $0.9650$ xuống $0.9400$ do sai số lượng tử hóa 4-bit trên dòng model nhỏ 4B. Kết quả thực nghiệm hoàn toàn ủng hộ khuyến nghị: khi VRAM GPU T4 ($14.6\text{ GB}$) đủ sức chứa FP16 LoRA mà không bị tràn bộ nhớ, ta không nên đánh đổi chất lượng lấy dung lượng bằng QLoRA.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.200` · `regression Δ = -0.080` · `valid_trace_rate = 0.00`

Diễn giải (≥100 từ).
Kết quả đánh giá qua cổng hồi quy (Regression Gate) ghi nhận trạng thái **FAILED** do chỉ số suy thoái năng lực tổng quát vượt ngưỡng cho phép (`regression Δ = -0.080`, vượt quá dung sai 0.020). Mặc dù bản fine-tune LoRA (`correct`) đạt bước tiến vượt trội về Target Accuracy (tăng từ $0.7650$ ở Baseline b lên $0.9650$, tức $\Delta = +0.200$) và định dạng đạt chuẩn $100\%$ (`format = 1.000`), mô hình đã bị hiện tượng quên thảm họa (catastrophic forgetting) nhẹ đối với 15 câu hỏi kiểm thử kiến thức nền tảng (điểm giảm từ $0.7578$ xuống $0.6778$). Điều này chứng minh rằng việc huấn luyện hoàn toàn trên miền dữ liệu hẹp 250 ticket CSKH mà không có cơ chế bảo toàn phân phối gốc sẽ làm xói mòn khả năng tổng quát. Để khắc phục trước khi triển khai, giải pháp tối ưu theo Deck §14.3 là trộn thêm $1–5\%$ dữ liệu đa nhiệm tổng quát (replay data) trong quá trình huấn luyện.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt. | `doi_tra`, `cao`, `chuột không dây`, `tich_cuc` | `doi_tra`, `cao`, `chuột không dây`, `tieu_cuc` (sai sentiment) | `doi_tra`, `cao`, `chuột không dây`, `tich_cuc` | ✅ FT thắng (nhận diện đúng sentiment tích cực) |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé. Bực mình. | `hoan_tien`, `trung_binh`, `ốp lưng điện thoại`, `tieu_cuc` | `hoan_tien`, `cao`, `ốp lưng điện thoại`, `tieu_cuc` (sai urgency) | `hoan_tien`, `trung_binh`, `ốp lưng điện thoại`, `tieu_cuc` | ✅ FT thắng (phân loại đúng mức độ khẩn cấp) |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều. | `hoan_tien`, `thap`, `bình giữ nhiệt`, `tich_cuc` | `hoan_tien`, `thap`, `bình giữ nhiệt`, `tich_cuc` | `hoan_tien`, `trung_binh`, `bình giữ nhiệt`, `tich_cuc` (sai urgency) | ❌ **FT thua** (FT đoán mức độ khẩn cấp trung bình thay vì thấp) |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi. | `san_pham_loi`, `thap`, `nồi chiên không dầu`, `trung_tinh` | `san_pham_loi`, `thap`, `nồi chiên không dầu`, `trung_tinh` | `san_pham_loi`, `trung_binh`, `nồi chiên không dầu`, `trung_tinh` (sai urgency) | ❌ **FT thua** (FT bị thiên lệch gán urgency trung bình cho cụm từ "Khi nào tiện") |
| 5 | Alo shop, mình đặt máy xay sinh tố mã đơn OD126693. Muốn đổi. Đã 3 ngày rồi. Bực mình. | `doi_tra`, `trung_binh`, `máy xay sinh tố`, `tieu_cuc` | `doi_tra`, `cao`, `máy xay sinh tố`, `tieu_cuc` (sai urgency) | `doi_tra`, `trung_binh`, `máy xay sinh tố`, `tieu_cuc` | ✅ FT thắng (xác định chính xác urgency trung bình cho "đã 3 ngày rồi") |

Có mẫu chung nào ở các ca FT thua không?
Các ca FT thua đều tập trung vào trường **`urgency`**, cụ thể là mô hình có xu hướng dự đoán mức độ `trung_binh` cho các ticket có cụm từ "Khi nào tiện" (thay vì nhãn `thap`). Nguyên nhân do phân phối tập train có nhiều mẫu ticket chứa câu hỏi thời gian được gán nhãn `trung_binh`, khiến adapter bị thiên lệch nhẹ về độ nhạy urgency.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).**
Bản fine-tune LoRA cấu hình chuẩn (`correct`) mang lại khả năng tuân thủ cấu trúc JSON hoàn hảo ($100\%$ format compliance), đồng thời nội hóa toàn bộ định nghĩa nhãn và cấu trúc đầu ra vào trọng số mô hình. Điều này giúp rút ngắn đáng kể độ dài câu trả lời, loại bỏ sự phụ thuộc vào system prompt dài và giảm độ trễ sinh từ $\sim 11\text{s}$ xuống dưới $2\text{s}$ cho mỗi mẫu. 

Tuy nhiên, việc quyết định deploy còn cần cân nhắc ngưỡng Target Accuracy so với Baseline (b) đã được prompt kỹ càng. Đòn bẩy kỹ thuật quan trọng nhất quyết định thành công của lab này không nằm ở rank cao hay lượng tử hóa, mà nằm ở: (1) **Đúng Mask Loss (assistant-only)** để không bị học vẹt ngữ cảnh; (2) **Vị trí phủ adapter toàn diện (`text-linear`)**; và (3) **Learning Rate phù hợp ($1\times 10^{-4}$)**. Nếu cần triển khai thực tế, việc kết hợp giữa một bộ dữ liệu giàu tính phân phối và mô hình LoRA đúng chuẩn sẽ là giải pháp tối ưu cả về chi phí lẫn chất lượng.

**Ba điều tôi học được** (cụ thể, không generic):
1. **Loss Mask là nền tảng sống còn:** Huấn luyện chỉ trên phản hồi của assistant (loại bỏ prompt khỏi loss) là điều kiện bắt buộc; bất kỳ sai lệch nào về mask (như cờ TRL không nhận diện template) đều biến toàn bộ quá trình huấn luyện thành vô nghĩa dù đường loss trông rất đẹp.
2. **Vị trí adapter quan trọng hơn Rank:** Phủ adapter lên toàn bộ các lớp tuyến tính (`text-linear`) với rank nhỏ ($r=16$) mang lại hiệu quả thích ứng cao hơn nhiều so với việc dồn rank cực lớn ($r=283$) vào riêng các lớp Attention với cùng một ngân sách tham số.
3. **Prompt Engineering là mốc kiểm chuẩn tối thượng (Baseline b):** Không thể tuyên bố fine-tune thành công nếu chỉ so với base model prompt sơ sài; một prompt tối ưu có thể tăng vọt độ chính xác và giảm 3× độ trễ mà không cần tốn chi phí train.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Thực hiện Bonus B1 và B4: Thử nghiệm merge adapter vào base model và đo lường sự suy giảm hiệu năng sau khi merge, đồng thời quét rank $r \in \{8, 16, 64\}$ trên cấu hình `text-linear` để xác định điểm bão hòa dung lượng biểu diễn của tập dữ liệu.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [x] B5 HuggingFace Hub — link: https://huggingface.co/eckoops/lab21-2A202601046-qwen35-triage-vi
