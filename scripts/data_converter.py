import json
import os
import argparse
from pathlib import Path

"""
Data Converter Utility — Converts JSON data to the JSONL format required by BookMind.
Standard format for each line in JSONL:
{
    "book_title": "Title of the Book",
    "title": "Chapter or Section Title",
    "content": "Text content..."
}
"""

def convert_json_to_jsonl(input_path, output_dir):
    """
    Reads a JSON file (or a directory of JSON files) and converts them to JSONL.
    
    Args:
        input_path: Path to a .json file or a directory containing .json files.
        output_dir: Directory where the .jsonl files will be saved.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print(f"📁 Created output directory: {output_dir}")

    # Identify files to process
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = list(input_path.glob("*.json"))
    else:
        print(f"❌ Path not found: {input_path}")
        return

    if not files:
        print(f"⚠️ No JSON files found in {input_path}")
        return

    print(f"🚀 Processing {len(files)} files...")

    for json_file in files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Ensure data is a list
            if not isinstance(data, list):
                data = [data]
            
            output_file = output_dir / f"{json_file.stem}.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for entry in data:
                    # Basic validation/cleanup can be added here
                    json_record = json.dumps(entry, ensure_ascii=False)
                    f_out.write(json_record + '\n')
            
            print(f"✅ Converted: {json_file.name} -> {output_file.name} ({len(data)} entries)")
            
        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert JSON to BookMind JSONL format")
    parser.add_argument("input", help="Input JSON file or directory")
    parser.add_argument("--output", default="data", help="Output directory (default: data)")
    
    args = parser.parse_args()
    convert_json_to_jsonl(args.input, args.output)

if __name__ == "__main__":
    main()
