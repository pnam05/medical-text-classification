"""Create a short Vietnamese discussion brief from the verified v4 artifacts.

Optional document dependencies: reportlab, pymupdf (not needed by the ML pipeline).
"""
from pathlib import Path
import json
import csv

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/tom_tat_tien_xu_li_va_huong_thay_the.pdf"
SOURCE = ROOT / "outputs/preprocessing"
summary = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
with (SOURCE / "tables/threshold_coverage.csv").open(encoding="utf-8", newline="") as stream:
    coverage = {int(row["min_class_groups"]): row for row in csv.DictReader(stream)}
r = summary['specialty_resolution']
n = summary['processed_rows']
k = len(summary['selected_specialties'])
sp = summary['split_rows']
low = r['resolved_usable_documents'] - n

pdfmetrics.registerFont(TTFont("ArialVN", "C:/Windows/Fonts/arial.ttf"))
pdfmetrics.registerFont(TTFont("ArialVN-Bold", "C:/Windows/Fonts/arialbd.ttf"))
pdfmetrics.registerFontFamily("ArialVN", normal="ArialVN", bold="ArialVN-Bold", italic="ArialVN", boldItalic="ArialVN-Bold")
NAVY = colors.HexColor("#163447")
TEAL = colors.HexColor("#167580")
GRAY = colors.HexColor("#526171")
styles = {
    "title": ParagraphStyle("title", fontName="ArialVN-Bold", fontSize=21, leading=26, textColor=NAVY, spaceAfter=8),
    "sub": ParagraphStyle("sub", fontName="ArialVN", fontSize=9, leading=13, textColor=GRAY, spaceAfter=12),
    "h": ParagraphStyle("h", fontName="ArialVN-Bold", fontSize=12, leading=16, textColor=TEAL, spaceBefore=12, spaceAfter=6),
    "body": ParagraphStyle("body", fontName="ArialVN", fontSize=10, leading=14.5, textColor=NAVY, spaceAfter=6),
    "cell": ParagraphStyle("cell", fontName="ArialVN", fontSize=9, leading=12.5, textColor=NAVY),
    "head": ParagraphStyle("head", fontName="ArialVN-Bold", fontSize=9, leading=12, textColor=colors.white),
    "note": ParagraphStyle("note", fontName="ArialVN", fontSize=8, leading=11, textColor=GRAY, spaceAfter=4),
}
story = []
def p(text, style="body"):
    story.append(Paragraph(text, styles[style]))

def table(headers, rows, widths):
    content = [[Paragraph(s, styles["head"]) for s in headers]]
    content += [[Paragraph(str(s), styles["cell"]) for s in row] for row in rows]
    obj = Table(content, colWidths=widths, repeatRows=1, hAlign="LEFT")
    obj.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f5f7"), colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#d5e0e6")),
    ]))
    story.append(obj)

p("Tiền xử lý MTSamples", "title")
p("Phạm vi 26 nhãn do nhóm chỉ định | Pipeline v4 | Chưa huấn luyện mô hình", "sub")
p("1. Công việc đã thực hiện", "h")
p("Gom bản gốc theo Unicode, chữ thường và khoảng trắng. Lấy hợp các nhãn trong phạm vi 26 nhãn trước lọc support. Đúng một nhãn mục tiêu: giữ một dòng nguồn mang nhãn đó; nhiều nhãn: tách để rà soát; không có nhãn: ngoài phạm vi. Không chọn nhãn đa số.")
p("Surgery + Urology được giữ một bản Urology; Urology + Nephrology vẫn phải rà soát dù một lớp bị loại bởi support. Nhãn ghép như Cardiovascular / Pulmonary được giữ nguyên. Danh sách lần này do nhóm xác định, không phải danh mục y tế chuẩn.")
table(["Bước", "Kết quả"], [
 ["Dòng nguồn / hợp lệ", f"{summary['raw_rows']} / {summary['catalog_rows']}"],
 ["Văn bản gốc độc lập", str(r['raw_groups'])],
 ["Đúng một nhãn mục tiêu / nhiều nhãn / ngoài phạm vi", f"{r['single_target_label']} / {r['ambiguous_target_labels']} / {r['no_target_label']}"],
 ["Văn bản dùng được sau kiểm tra cleaner", str(r['resolved_usable_documents'])],
 ["Benchmark / lớp ít mẫu lưu riêng", f"{n} văn bản, {k} lớp / {low} văn bản"],
 ["Train / validation / test", f"{sp['train']} / {sp['validation']} / {sp['test']}"],
], [275, 236])
p("2. Các lớp đủ điều kiện đánh giá", "h")
table(["Chuyên khoa", "Văn bản độc lập"], [[x['medical_specialty'], x['independent_groups']] for x in summary['class_support'] if x['selected']], [360,151])
p("Ngưỡng giữ nguyên: ít nhất 100 nhóm độc lập/lớp; kiểm tra tối thiểu 20 nhóm/lớp ở từng holdout. Không giới hạn Top K. 26 nhãn được xét không có nghĩa cả 26 lớp đều đủ mẫu.", "note")
p("Bàn giao: văn bản gốc và bản sạch bảo thủ (giữ stopwords, số đo, phủ định), nhãn số, split/CV theo nhóm, TF-IDF chỉ fit train, dấu vết loại dòng, hàng đợi rà soát và 18 biểu đồ. Catalog giữ mọi dòng dùng được; không gộp catalog vào train vì có cả holdout.")
story.append(PageBreak())
p("Hạn chế và hướng thay thế", "title")
p("3. Hạn chế", "h")
p(f"<b>Quy mô:</b> chỉ {sp['train']} văn bản train. Ngưỡng 100 không phải tối ưu đã chứng minh; support tối thiểu không bảo đảm F1 ổn định. Cần báo cáo số mẫu từng lớp và độ bất định.")
p("<b>Phạm vi:</b> chỉ đánh giá ca đơn nhãn thuộc các lớp đủ support. Bỏ ca đa chuyên khoa có thể làm bài toán dễ hơn. Không suy rộng kết quả cho toàn bộ 40 nhãn hoặc so trực tiếp điểm với phiên bản dữ liệu khác.")
p("<b>Chất lượng nhãn:</b> nhất quán nhãn không chứng minh nhãn đúng lâm sàng. Trùng khác nhãn chưa chắc là lỗi vì một bệnh án có thể thuộc nhiều chuyên khoa. Chưa dò trùng gần hoặc cách ly theo patient ID; chưa có accuracy/F1.")
p("4. Các phương án thay thế", "h")
table(["Hướng", "Lợi ích và đánh đổi"], [
 ["Hạ ngưỡng support", f"Ngưỡng 50 giữ {coverage[50]['rows']} văn bản thuộc {coverage[50]['classes']} lớp trước kiểm tra split. Tăng độ bao phủ nhưng ít mẫu holdout/lớp hơn; phải chia và kiểm tra lại."],
 ["Duyệt nhãn thủ công", f"Rà soát {r['ambiguous_target_labels']} nhóm nhiều nhãn. Viết hướng dẫn, gán nhãn độc lập và phân xử bất đồng; ưu tiên chuyên môn phù hợp. Tốn công, không bảo đảm mọi ca đều có một nhãn đúng."],
 ["Phân loại đa nhãn", "Phù hợp nếu nhiều chuyên khoa cùng đúng. Cần kiểm chứng phạm vi và độ đầy đủ của annotation; hợp nhãn nguồn chưa phải ground truth đầy đủ. Đổi mô hình và metric sang đa nhãn."],
 ["Phân loại phân cấp", "Biểu diễn nhãn rộng/hẹp nhưng cần duyệt quan hệ nhãn. Không tự gộp lớp chỉ để tăng mẫu; chưa giải quyết thiếu mẫu hoặc mơ hồ nếu chưa duyệt nhãn."],
 ["Bổ sung hoặc đổi dataset", "Hữu ích khi cần so sánh ML-DL với quy mô lớn hơn. Kiểm tra quyền sử dụng, tương thích nhãn, khác biệt nguồn và trùng lặp trước khi gộp."],
], [150,361])
p("5. Đề xuất cho nhóm", "h")
p("Giữ cấu hình hiện tại làm baseline, thử ngưỡng 50 như phân tích đánh đổi độ bao phủ. Nếu trọng tâm là nhãn nhiễu, ưu tiên duyệt nhóm mơ hồ; chuyển đa nhãn khi nhiều nhãn cùng đúng. Có thể giữ test khoảng 20% theo nhóm và dùng phần còn lại cho grouped CV để tận dụng dữ liệu, nhưng không tạo thêm bệnh án độc lập.")
p("Chỉ dùng train/CV để chọn cấu hình. Đổi phạm vi hoặc nhãn phải tạo phiên bản dữ liệu mới. Không tuyên bố đã sửa toàn bộ nhãn nhiễu; các hướng thay thế chưa được huấn luyện hoặc đo hiệu năng trong báo cáo này.", "note")
p("Nguồn: outputs/preprocessing/summary.json; tables/class_eligibility.csv; tables/threshold_coverage.csv; configs/specialty_scope.json.", "note")

def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor("#d5e0e6"))
    canvas.line(42, 35, A4[0]-42, 35)
    canvas.setFont("ArialVN", 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(42, 23, "MTSamples | Tài liệu thảo luận nhóm | Pipeline v4")
    canvas.drawRightString(A4[0]-42, 23, str(doc.page))

OUT.parent.mkdir(parents=True, exist_ok=True)
SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=42, leftMargin=42,
                  topMargin=34, bottomMargin=46,
                  title="Tóm tắt tiền xử lý MTSamples và hướng thay thế", author="Nhóm đồ án").build(
    story, onFirstPage=footer, onLaterPages=footer)

import pymupdf as fitz
document = fitz.open(OUT)
render_dir = ROOT / "tmp/pdfs/preprocessing_brief"
render_dir.mkdir(parents=True, exist_ok=True)
for index, page in enumerate(document):
    page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(render_dir / f"page-{index+1}.png")
text = "\n".join(page.get_text() for page in document)
assert str(n) in text and "26 nhãn" in text and "Hạn chế" in text
assert len(document) == 2, f"Expected 2 pages, got {len(document)}; review layout."
print(f"Created {OUT}; {len(document)} pages. Rendered to {render_dir}")
