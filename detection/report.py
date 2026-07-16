import json
import os
from datetime import datetime


def generate_detection_report(result, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{result['result_id']}_report.pdf")
    lines = [
        "Quantum-Assisted Steganography Detection Report",
        f"Image name: {result['image_name']}",
        f"Date and time: {result.get('created_at') or datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Prediction: {result['prediction']}",
        f"Confidence score: {result['confidence']}%",
        f"Quantum model used: {result.get('model', 'Qiskit Quantum Classifier')}",
        f"Processing time: {result['processing_time']}",
        "Extracted features:",
    ]
    lines.extend(f"  - {item['label']}: {item['value']}" for item in result.get("feature_summary", []))

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        canvas_doc = canvas.Canvas(report_path, pagesize=letter)
        width, height = letter
        y = height - 72
        canvas_doc.setFont("Helvetica-Bold", 15)
        canvas_doc.drawString(72, y, lines[0])
        y -= 32
        canvas_doc.setFont("Helvetica", 10.5)
        for line in lines[1:]:
            if y < 72:
                canvas_doc.showPage()
                y = height - 72
                canvas_doc.setFont("Helvetica", 10.5)
            canvas_doc.drawString(72, y, line)
            y -= 18
        canvas_doc.save()
    except ImportError:
        write_minimal_pdf(report_path, lines)

    json_path = os.path.join(output_dir, f"{result['result_id']}_report.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    return report_path


def write_minimal_pdf(path, lines):
    content = ["BT", "/F1 14 Tf", "72 750 Td"]
    for index, line in enumerate(lines):
        size = 14 if index == 0 else 10
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content.append(f"/F1 {size} Tf")
        content.append(f"({escaped}) Tj")
        content.append("0 -18 Td")
    content.append("ET")
    stream = "\n".join(content).encode("latin-1", errors="replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode("ascii")
    )

    with open(path, "wb") as handle:
        handle.write(output)
