import mimetypes
from pathlib import Path
from pembot.pdf2markdown.extract import MarkdownPDFExtractor
import os


EXCEL_FILE_TYPES= [
        'application/vnd.ms-excel',
        'application/msexcel',
        'application/x-msexcel',
        'application/x-ms-excel',
        'application/x-excel',
        'application/x-dos_ms_excel',
        'application/x-dos_ms_excel',
        'application/xls',
        'application/x-xls',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
]

class Convertor():
    def __init__(self, myfile: Path, output_path: Path):
        print("got output path for conversion: ", output_path)
        mt= mimetypes.guess_file_type(str(myfile))[0]
        if mt == 'application/json':
            print("the file was json")
        elif mt == 'application/pdf':
            print("the file was pdf, outputting in: ", output_path)
            extractor= MarkdownPDFExtractor(str(myfile), output_path= str(output_path))
            extractor.extract()

            base_name, _ = os.path.splitext(myfile.name)
            new_filename = base_name + ".md"

        elif mt == 'text/csv':
            print("the file was csv")
        elif mt in EXCEL_FILE_TYPES:
            print("the file was excel spreadsheet")
        elif mt == 'application/vnd.oasis.opendocument.spreadsheet':
            print("the file was opentype spreadsheet")
        else:
            print(mt)


def chunk_text(text, chunk_size=500, overlap_size=50):
    """
    Chunks a given text into smaller pieces with optional overlap.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size of each chunk (in characters).
        overlap_size (int): The number of characters to overlap between consecutive chunks.

    Returns:
        list: A list of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap_size)
        if start < 0:  # Handle cases where overlap_size is greater than chunk_size
            start = 0
    return chunks

if __name__ == '__main__':
    print("do you want a rice bag?")

