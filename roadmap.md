# 0) Chốt phạm vi hệ thống trước khi code

## Việc cần làm

- Chốt 4 use case:
    - chitchat
    - appointment
    - simple_qa
    - complex_qa
- Chốt domain dữ liệu:
    - thuốc
    - dược liệu
    - bệnh
    - hiểu về cơ thể
- Chốt dạng câu trả lời:
    - trả lời ngắn
    - trả lời có nguồn
    - có cảnh báo an toàn khi cần

## Expected output

- Một tài liệu scope ngắn gọn
- Danh sách use case và route tương ứng
- Danh sách nguồn dữ liệu sẽ crawl
- Tiêu chí thành công ban đầu, ví dụ:
    - router phân loại đúng
    - simple QA trả lời có căn cứ
    - complex QA có multi-step retrieval
    - appointment gọi được tool mô phỏng

---

# 1) Thu thập dữ liệu

## Việc cần làm

- Xác định nguồn chính thống và ổn định
- Crawl theo từng nhóm:
    - thuốc
    - dược liệu
    - bệnh
    - cơ thể / giải phẫu / sinh lý
- Lưu cả metadata:
    - title
    - url
    - source
    - crawl time
    - category
    - raw html/text

## Expected output

- `data/raw/*.jsonl` hoặc `data/raw/*.parquet`
- Mỗi record có cấu trúc rõ ràng, ví dụ:

```json
{
  "id": "...",
  "title": "...",
  "content_raw": "...",
  "url": "...",
  "source": "...",
  "category": "drug",
  "crawl_time": "..."
}
```

## Điều kiện hoàn thành

- Dữ liệu đủ lớn để thử nghiệm
- Không bị trùng quá nhiều
- Có thể trace lại nguồn gốc từng đoạn nội dung

---

# 2) Làm sạch và chuẩn hóa dữ liệu

## Việc cần làm

- Loại bỏ HTML rác, menu, footer, quảng cáo
- Chuẩn hóa encoding tiếng Việt
- Loại bản ghi lỗi / quá ngắn / trùng lặp
- Chuẩn hóa field:
    - title
    - content
    - category
    - source
    - url
- Có thể gắn thêm nhãn ngữ nghĩa nếu cần

## Expected output

- `data/processed/*.jsonl` hoặc `.parquet`
- Dữ liệu sạch, nhất quán, dùng được cho embedding

## Điều kiện hoàn thành

- Văn bản không còn rác HTML đáng kể
- Không còn record trống
- Có thể đọc trực tiếp để chunking

---

# 3) Chunking và embedding

## Việc cần làm

- Chia văn bản thành chunk hợp lý
- Mỗi chunk nên giữ ngữ cảnh vừa đủ
- Gắn metadata cho mỗi chunk:
    - doc_id
    - section
    - category
    - source
- Tạo embedding cho chunk

## Expected output

- Bộ chunks đã chuẩn hóa
- Vector embeddings
- File/index phục vụ retrieval

Ví dụ:

- `data/processed/chunks.parquet`
- `data/embeddings/index.faiss`
- hoặc collection trong vector DB

## Điều kiện hoàn thành

- Mỗi chunk có thể truy xuất độc lập
- Không bị chunk quá dài hoặc quá ngắn
- Retrieval có thể trả về đúng chunk liên quan

---

# 4) Xây retrieval baseline trước

## Việc cần làm

- Làm retriever đơn giản trước:
    - dense retrieval
    - hoặc hybrid retrieval nếu đủ thời gian
- Test theo từng loại query:
    - thuốc
    - dược liệu
    - bệnh
    - cơ thể

## Expected output

- Một hàm hoặc service:

```python
retrieve(query) -> top_k_chunks
```

- Có thể in ra top-k kết quả liên quan
- Có citation/source đi kèm

## Điều kiện hoàn thành

- Query đơn giản trả về đúng nội dung
- Có thể dùng làm nền cho RAG

---

# 5) Xây baseline RAG “thuần”

## Việc cần làm

- Flow cơ bản:

```
Query → Retriever → Synthesizer → Answer
```

- Prompt sinh câu trả lời phải:
    - bám context
    - tránh bịa
    - ưu tiên nguồn truy xuất
- Có output dạng dễ kiểm tra

## Expected output

- Một endpoint hoặc function trả lời câu hỏi dựa trên dữ liệu
- Câu trả lời có thể trích nguồn / context
- Baseline để so sánh với agentic version sau này

## Điều kiện hoàn thành

- Simple QA chạy ổn
- Không cần agent phức tạp vẫn trả lời được

---

# 6) Thiết kế Router

## Việc cần làm

- Xác định label routing:
    - chitchat
    - appointment
    - simple_qa
    - complex_qa
- Viết rule-based router trước hoặc LLM router
- Đầu ra router nên là JSON có cấu trúc rõ ràng, ví dụ:

```json
{
  "route": "complex_qa",
  "complexity": "high",
  "need_tool": false
}
```

## Expected output

- Router node/function
- Bộ test câu hỏi cho router
- Tỉ lệ phân loại đúng tương đối ổn

## Điều kiện hoàn thành

- Query vào là biết đi nhánh nào
- Router đủ ổn để phân nhánh pipeline

---

# 7) Thiết kế Planner

## Việc cần làm

- Planner chỉ dùng cho:
    - appointment
    - complex_qa
- Với appointment:
    - trích xuất thông tin đặt lịch
- Với complex_qa:
    - tách câu hỏi thành nhiều bước retrieval
- Output nên là plan có cấu trúc, không chỉ là text tự do

Ví dụ:

```json
{
  "steps": [
    "retrieve drug info",
    "retrieve disease info",
    "check interaction",
    "synthesize answer"
  ]
}
```

## Expected output

- Planner node
- JSON plan cho từng loại query phức tạp
- Có thể dùng lại trong LangGraph

## Điều kiện hoàn thành

- Query nhiều ý được tách thành các tiểu nhiệm vụ hợp lý
- Planner không tạo plan rối hoặc thừa bước

---

# 8) Thiết kế các node trong LangGraph

## Việc cần làm

Thiết kế tối thiểu các node sau:

- `router`
- `planner_appointment`
- `planner_complex`
- `simple_retriever`
- `multi_retriever`
- `tool_appointment`
- `synthesizer`
- `evaluator`

## Expected output

- Graph node-by-node rõ ràng
- State schema thống nhất
- Có conditional edge giữa các nhánh

## Điều kiện hoàn thành

- Compile được graph
- Chạy được cả 4 flow:
    - chitchat
    - appointment
    - simple_qa
    - complex_qa

---

# 9) Thiết kế backend

## Việc cần làm

- Dùng FastAPI làm backend
- Tạo các API chính:
    - `POST /chat`
    - `POST /appointment`
    - `GET /health`
    - `GET /sources/{id}` nếu cần debug nguồn
- Backend gọi LangGraph pipeline
- Trả về:
    - answer
    - route
    - citations/context
    - tool result nếu có

## Expected output

- API server chạy được
- Có Swagger UI
- Frontend có thể gọi backend

## Điều kiện hoàn thành

- Request từ ngoài vào xử lý end-to-end
- Có logging để debug từng bước

---

# 10) Thiết kế frontend

## Việc cần làm

- Dựng UI tối thiểu:
    - ô chat
    - vùng hiển thị câu trả lời
    - hiển thị route / trạng thái xử lý nếu cần
    - hiển thị citation/source
- Nếu có appointment:
    - form đặt lịch hoặc message-based booking
- Nếu muốn đẹp hơn:
    - hiển thị “thinking steps”
    - hiển thị kết quả retriever
    - hiển thị trạng thái tool

## Expected output

- Giao diện người dùng có thể nhập câu hỏi và nhận câu trả lời
- Có thể demo các use case trực tiếp
- Nhìn đủ “end-to-end system”

## Điều kiện hoàn thành

- Frontend gọi backend thành công
- Có trải nghiệm chat mượt
- Có hiển thị trạng thái hợp lý

---

# 11) Tích hợp end-to-end

## Việc cần làm

- Nối frontend → backend → LangGraph → retrieval/tool → response
- Kiểm tra từng flow:
    - chitchat
    - appointment
    - simple_qa
    - complex_qa
- Đồng bộ format response

## Expected output

- Hệ thống hoàn chỉnh chạy từ UI đến logic xử lý
- Có thể demo ngay trên trình duyệt

## Điều kiện hoàn thành

- Không còn mock rời rạc
- Dữ liệu thật và logic thật đã nối với nhau

---

# 12) Deploy

## Việc cần làm

- Dockerize backend
- Dockerize frontend
- Tạo `docker-compose.yml`
- Nếu có vector DB / Redis / database thì container hóa luôn
- Deploy lên server/cloud

## Expected output

- Một lệnh là có thể chạy toàn bộ hệ thống
- Có môi trường production-like
- Demo ổn định hơn chạy local

## Điều kiện hoàn thành

- Chạy được sau khi pull repo
- Không phụ thuộc vào máy cá nhân
- Có `.env` rõ ràng

---

# 13) Test

## Việc cần làm

Tách test thành 4 lớp:

### 1. Unit test

- Router đúng chưa
- Planner đúng JSON chưa
- Retriever trả context đúng chưa
- Tool appointment chạy đúng chưa

### 2. Integration test

- Router → Planner → Retriever → Synthesizer
- Backend gọi graph có trả kết quả không

### 3. End-to-end test

- Từ frontend nhập câu hỏi đến nhận kết quả cuối

### 4. Quality evaluation

- Độ đúng của router
- Độ liên quan retrieval
- Tính nhất quán của câu trả lời
- Tính đầy đủ của complex QA

## Expected output

- Bộ test case rõ ràng
- Có log pass/fail
- Có số liệu đánh giá để đưa vào report

## Điều kiện hoàn thành

- Hệ thống không chỉ “chạy được” mà còn “được kiểm chứng”

---

# 14) Đánh giá chất lượng cuối cùng

## Việc cần làm

- Tạo bộ câu hỏi benchmark
- Chia theo 4 use case
- Ghi nhận:
    - router accuracy
    - retrieval relevance
    - answer groundedness
    - tool success rate
- So sánh baseline RAG với agentic pipeline

## Expected output

- Bảng kết quả đánh giá
- Biểu đồ so sánh trước/sau
- Kết luận rõ ràng: agentic pipeline cải thiện gì

## Điều kiện hoàn thành

- Có số liệu để viết báo cáo
- Có thể chứng minh giá trị của kiến trúc mới

---

# Trình tự thực hiện thực tế nên là như sau

Nếu bạn muốn đi đúng nhịp và tránh làm rối, thứ tự tối ưu là:

1. Chốt scope và use case
2. Crawl dữ liệu
3. Làm sạch và chuẩn hóa
4. Chunk + embedding
5. Làm baseline RAG
6. Xây Router
7. Xây Planner
8. Thiết kế LangGraph nodes
9. Làm backend FastAPI
10. Làm frontend
11. Tích hợp end-to-end
12. Deploy bằng Docker
13. Viết test và benchmark
14. Đánh giá và hoàn thiện báo cáo

---

# Mốc “xong” của từng giai đoạn

## Giai đoạn dữ liệu

Xong khi:

- có dataset sạch, có thể truy xuất nguồn

## Giai đoạn retrieval

Xong khi:

- simple question trả đúng context

## Giai đoạn routing/planning

Xong khi:

- query được phân nhánh đúng
- complex query được tách bước đúng

## Giai đoạn backend/frontend

Xong khi:

- người dùng có thể chat và nhận phản hồi thật

## Giai đoạn deploy/test

Xong khi:

- hệ thống chạy được độc lập
- có số liệu đánh giá rõ ràng
