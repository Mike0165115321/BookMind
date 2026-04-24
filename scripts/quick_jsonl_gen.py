import json
import os
import re
from pathlib import Path
from collections import defaultdict

"""
Universal JSONL Generator — Processes both Template and JSON raw strings.
Groups entries by book_title into a single .jsonl file per book.
"""

def parse_list_field(text, pattern):
    match = re.search(pattern, text)
    if not match:
        return []
    items = match.group(1).split(",")
    return [item.strip() for item in items if item.strip()]

def parse_field(text, pattern, default=""):
    match = re.search(pattern, text)
    return match.group(1).strip() if match else default

def save_grouped_data(grouped_data, output_dir):
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    for book_title, entries in grouped_data.items():
        # Generate filename from book title
        safe_book_name = re.sub(r'[^\w\s-]', '', book_title).strip().replace(' ', '_')
        file_path = output_path / f"{safe_book_name}.jsonl"

        with open(file_path, "w", encoding="utf-8") as f_out:
            for entry in entries:
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        print(f"✅ รวมไฟล์สำเร็จ: {file_path} (รวม {len(entries)} หัวข้อ)")

def process_file(file_path, output_dir):
    if not os.path.exists(file_path):
        print(f"❌ ไม่พบไฟล์ {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Data structure to group by book_title
    grouped_data = defaultdict(list)

    # Check if the file is a series of JSON objects
    is_json_format = False
    for line in lines:
        if line.strip().startswith("{"):
            is_json_format = True
            break

    if is_json_format:
        print("🔍 ตรวจพบรูปแบบ JSON — กำลังรวบรวมข้อมูล...")
        for i, line in enumerate(lines):
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                book_title = data.get("book_title") or "Unknown_Book"
                grouped_data[book_title].append(data)
            except Exception as e:
                print(f"⚠️ บรรทัดที่ {i+1} ไม่ใช่ JSON ที่ถูกต้อง: {e}")
    else:
        # Process as Template
        print("🔍 ตรวจพบรูปแบบ Template — กำลังดำเนินการ...")
        raw_text = "".join(lines)
        try:
            data = {
                "book_title": parse_field(raw_text, r"BOOK_TITLE:\s*(.*)"),
                "category": parse_field(raw_text, r"CATEGORY:\s*(.*)", "ระบบชีวิต"),
                "chapter_title": parse_field(raw_text, r"CHAPTER_TITLE:\s*(.*)"),
                "subsection_title": parse_field(raw_text, r"SUBSECTION_TITLE:\s*(.*)"),
                "title": parse_field(raw_text, r"TITLE:\s*(.*)"),
                "description": parse_field(raw_text, r"DESCRIPTION:\s*(.*)"),
                "strategy_type": parse_field(raw_text, r"STRATEGY_TYPE:\s*(.*)"),
                "influence_level": parse_field(raw_text, r"INFLUENCE_LEVEL:\s*(.*)"),
                "adaptability_level": parse_field(raw_text, r"ADAPTABILITY_LEVEL:\s*(.*)"),
                "psychological_techniques": parse_list_field(raw_text, r"PSYCHOLOGICAL_TECHNIQUES:\s*(.*)"),
                "risk_factors": parse_list_field(raw_text, r"RISK_FACTORS:\s*(.*)"),
                "control_techniques": parse_list_field(raw_text, r"CONTROL_TECHNIQUES:\s*(.*)"),
            }

            content_match = re.search(r"---CONTENT_START---(.*)---CONTENT_END---", raw_text, re.DOTALL)
            if content_match:
                data["content"] = content_match.group(1).strip()
                book_title = data.get("book_title") or "Unknown_Book"
                grouped_data[book_title].append(data)
            else:
                print("❌ ไม่พบ Marker ---CONTENT_START--- หรือ ---CONTENT_END---")
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาด: {e}")

    # Save all grouped data
    if grouped_data:
        save_grouped_data(grouped_data, output_dir)

if __name__ == "__main__":
    process_file("content_to_process.txt", "data")
