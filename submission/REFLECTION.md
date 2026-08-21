# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Sự chênh lệch giữa prompt sơ sài và prompt tối ưu ở Baseline: prompt tối ưu (b) không chỉ đạt độ chính xác cao gấp nhiều lần (từ 0.00 lên ~0.76) mà còn giúp giảm độ trễ ~3× (từ 11s xuống ~3.7s/prompt) do model dừng lại ngay sau khi xuất đủ JSON thay vì sinh lan man. Ngoài ra, việc TRL gặp lỗi ngầm khi `assistant_only_loss=True` không tìm thấy marker trong template dẫn đến train với mask rỗng (0% token tính loss) mà không hề raise exception là một bất ngờ lớn về độ nguy hiểm của silent bug.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Mất nhiều thời gian nhất ở công đoạn sinh văn bản (text generation / inference) khi đo 3 baseline ở NB2 và 4 lượt eval ở NB5, chứ không phải ở bước tính gradient/backward. Việc tập eval phải được sinh 3 lần khiến thời gian sinh chiếm hơn 60% tổng thời gian pipeline trên GPU T4 (~100–130 phút).

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước đây tôi tin rằng: (1) Cứ fine-tune là đương nhiên vượt trội hơn Prompt Engineering; và (2) Train loss càng thấp thì chất lượng mô hình càng cao. Lab này chứng minh prompt chuẩn chỉnh là một mốc rất cao và khó vượt qua nếu chỉ train vài chục step, và việc xếp hạng model bằng train loss (Lỗi #3) là sai lầm vì run `attn_only` có loss thấp hơn `correct` nhưng điểm target thực tế lại kém hơn.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi dùng AI assistant để kiểm tra cấu trúc pipeline, rà soát unit test, kiểm tra mask proof và xử lý các vấn đề môi trường cục bộ (chuẩn hóa LF/CRLF cho checksum). Chỗ AI hay mắc bẫy là thói quen áp đặt `bf16=True` cho mọi kiến trúc mà quên rằng Tesla T4 (Turing) chỉ hỗ trợ fp16, và nhầm lẫn giữa train loss với metric đánh giá thực tế.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Đóng băng một tập đánh giá chuẩn (Golden Eval Set) đại diện đúng phân phối nghiệp vụ, và xây dựng một Baseline Prompt thật mạnh (có schema, vài ví dụ one/few-shot) để đo đạc 4 nhóm chỉ số (Target, Regression, Format, Latency). Chỉ khi prompt engineering chạm trần giới hạn context, latency hoặc độ chính xác thì mới bắt tay vào thu thập dữ liệu và fine-tune.

