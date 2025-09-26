from tempfile import TemporaryDirectory
import mimetypes
from pathlib import Path
from pembot.pdf2markdown.extract import MarkdownPDFExtractor
import os
import pandas as pd
from typing import Literal, Union
import tempfile
from datetime import datetime, date
from tabulate import tabulate


PandasReadEngineType = Literal['xlrd', 'openpyxl', 'odf', 'pyxlsb', 'calamine', None]

EXCEL_FILE_TYPES= [
        'text/csv',
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
        'application/vnd.oasis.opendocument.spreadsheet',
]


class Convertor():


    def __init__(self, myfile: Path | None= None, output_dir: Path | None= None, file_bytes: bytes | None= None, suffix: str | None= None, file_type: str | None= None, model_name: str | None = None):

        self.output= ""
        self.suffix= suffix


        if model_name is None:
            # model_name=  "gemini-2.5-flash"
            model_name=  "Nanonets-OCR-s"

        # file_type can be pdf, excel, etc.
        if file_bytes and suffix:
            with tempfile.TemporaryDirectory() as dp:
                output_dir = Path(dp)
                myfile = output_dir / f"input{suffix}"
                myfile.write_bytes(file_bytes)

                if file_type == 'pdf':
                    extractor = MarkdownPDFExtractor(str(myfile), output_path=str(output_dir), page_delimiter="-- NEXT PAGE --", model_name=model_name)
                    extractor.extract()
                    with open(output_dir / (myfile.stem + '.md')) as output_file:
                        self.output = output_file.read()
                elif file_type == 'excel':
                    self.input_filepath = myfile
                    self.output = self.convert_excel_to_markdown()


        elif output_dir is not None and myfile is not None:
            print("got output path for conversion: ", output_dir)
            mt= mimetypes.guess_file_type(str(myfile))[0]

            self.output_dir= output_dir
            self.input_filepath= myfile

            if mt == 'application/json':
                print("the file was json")
            elif mt == 'application/pdf':
                print("the file was pdf, outputting in: ", output_dir)
                extractor= MarkdownPDFExtractor(str(myfile), output_path= str(self.output_dir), page_delimiter= "-- NEXT PAGE --", model_name= model_name)
                extractor.extract()
                with open(self.output_dir / (myfile.stem + '.md')) as output_file:
                    self.output= output_file.read()

            elif mt in EXCEL_FILE_TYPES:
                self.output = self.convert_excel_to_markdown()
                if myfile and output_dir:
                    with open(output_dir / (myfile.stem + '.md'), "w") as output_file:
                        output_file.write(self.output)

            else:
                print(mt)

    def convert_excel_to_markdown(self, excel_ods_engine: PandasReadEngineType = None) -> str:
        """
        Converts all sheets from an Excel or ODS file into a single Markdown string.
        Each sheet is converted to a Markdown table, prefixed with the sheet's name.

        Args:
            excel_ods_engine (str | None, optional): Pandas engine for reading Excel or ODS files.
                - For Excel: 'openpyxl' (for .xlsx), 'xlrd' (for .xls).
                - For ODS: 'odf' (requires 'odfpy' library).
                If None, pandas auto-detects based on file extension and installed libraries.

        Returns:
            str: A string containing the Markdown tables for all sheets, or an error message.
        """
        input_filepath = self.input_filepath
        markdown_output = []

        file_suffix= ''
        try:
            if not input_filepath.exists():
                file_suffix= self.suffix
            else:
                file_suffix = input_filepath.suffix.lower()

            current_engine: PandasReadEngineType = excel_ods_engine

            if file_suffix in ['.xls', '.xlsx', '.ods']:
                if file_suffix == '.ods':
                    if current_engine is None:
                        current_engine = 'odf'
                    elif current_engine != 'odf':
                        print(f"Warning: Specified engine '{current_engine}' may not be optimal for ODS. Forcing 'odf'.")
                        current_engine = 'odf'

                excel_file = pd.ExcelFile(input_filepath, engine=current_engine)
                if not excel_file.sheet_names:
                    return f"Warning: File '{input_filepath.name}' contains no sheets."

                for sheet_name in excel_file.sheet_names:
                    df = excel_file.parse(sheet_name)
                    markdown_output.append(f"## {sheet_name}\n")
                    markdown_table = tabulate(df, headers='keys', tablefmt='pipe')
                    markdown_output.append(markdown_table)
                    markdown_output.append("\n")

                return "\n".join(markdown_output)

            elif file_suffix == '.csv':
                df = pd.read_csv(input_filepath)
                markdown_table = tabulate(df, headers='keys', tablefmt='pipe')
                return markdown_table

            else:
                return f"Error: Unsupported file type: '{file_suffix}'. Please provide a CSV, XLS, XLSX, or ODS file."

        except ImportError as ie:
            if 'odfpy' in str(ie).lower() and file_suffix == '.ods':
                return f"Error reading ODS file '{input_filepath.name}': The 'odfpy' library is required. Please install it using 'pip install odfpy'."
            elif 'xlrd' in str(ie).lower() and file_suffix == '.xls':
                return f"Error reading .xls file '{input_filepath.name}': The 'xlrd' library might be required. Please install it using 'pip install xlrd'."
            elif 'openpyxl' in str(ie).lower() and file_suffix == '.xlsx':
                return f"Error reading .xlsx file '{input_filepath.name}': The 'openpyxl' library might be required. Please install it using 'pip install openpyxl'."
            else:
                return f"ImportError reading file '{input_filepath.name}': {ie}"
        except Exception as e:
            return f"An unexpected error occurred during conversion of '{input_filepath.name}': {e}"


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
    print("Test Run Start:")
    try:
        # print("Test 1: scaned pdf page, bytes")
        # with open("/home/cyto/Documents/scanned.pdf", "rb") as imgpdf:
        #     conv= Convertor(file_bytes= imgpdf.read(), suffix= ".pdf", file_type= "pdf")
        #     print(conv.output)

        # print("Test 2: JD pdf, bytes")
        # with open("/home/cyto/dev/pembotdir/jds/PM Trainee.pdf", "rb") as imgpdf:
        #     conv= Convertor(file_bytes= imgpdf.read(), suffix= ".pdf", file_type= "pdf")
        #     print(conv.output)

        print("Test 3: excel schedule, bytes")
        with open("/home/cyto/Downloads/Assignment schedule.xlsx", "rb") as imgpdf:
            conv= Convertor(file_bytes= imgpdf.read(), suffix= ".xlsx", file_type= "excel")
            print(conv.output)

        # without bytes example:
        print("Test 4: scanned pdf, path")
        conv= Convertor(myfile= Path('/home/cyto/Documents/scanned.pdf'), output_dir= Path('/home/cyto/Documents'))
        print(conv.output)

        print("Test 5: schedule excel, path")
        conv= Convertor(myfile= Path('/home/cyto/Downloads/Assignment schedule.xlsx'), output_dir= Path('/home/cyto/Downloads'))
        print(conv.output)
    except FileNotFoundError as fe:
        print("file not found, modify the driver code to get sample files to test:\n\n", fe)
    except Exception as e:
        print("unhandled: ", e)

    print("Test Run End.")
