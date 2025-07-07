from tempfile import TemporaryDirectory
import mimetypes
from pathlib import Path
from pembot.pdf2markdown.extract import MarkdownPDFExtractor
import os
import json
import pandas as pd
from typing import Literal, Union, Dict, Any, List
import tempfile
from datetime import datetime, date


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

        if model_name is None:
            # model_name=  "gemini-2.5-flash"
            model_name=  "Nanonets-OCR-s"

        # file_type can be pdf, excel, etc.
        if output_dir is None and myfile is None and file_bytes is not None and suffix is not None:
            with tempfile.TemporaryDirectory() as dp:
                with tempfile.NamedTemporaryFile(suffix= suffix, mode= 'wb') as fp:
                    fp.write(file_bytes)
                    myfile= Path(fp.name)
                    output_dir= Path(dp)
                    if file_type == 'pdf':
                        extractor= MarkdownPDFExtractor(str(myfile), output_path= str(output_dir), page_delimiter= "-- NEXT PAGE --", model_name= model_name)
                        extractor.extract()
                        with open(output_dir / (myfile.stem + '.md')) as output_file:
                            self.output= output_file.read()
                    elif file_type == 'excel':
                        self.input_filepath= myfile
                        self.json_filepath = output_dir / (myfile.stem + ".json")
                        self.convert_file_to_json()
                        with open(output_dir / (myfile.stem + '.json')) as output_file:
                            self.output= output_file.read()

        elif output_dir is not None and myfile is not None:
            print("got output path for conversion: ", output_dir)
            mt= mimetypes.guess_file_type(str(myfile))[0]

            self.output_dir= output_dir
            self.input_filepath= myfile
            base_name, _ = os.path.splitext(myfile.name)
            self.json_filepath = output_dir / 'json' / (base_name + ".json")

            if mt == 'application/json':
                print("the file was json")
            elif mt == 'application/pdf':
                print("the file was pdf, outputting in: ", output_dir)
                extractor= MarkdownPDFExtractor(str(myfile), output_path= str(self.output_dir), page_delimiter= "-- NEXT PAGE --", model_name= model_name)
                extractor.extract()

            elif mt in EXCEL_FILE_TYPES:
                self.convert_file_to_json()

            else:
                print(mt)

    def convert_file_to_json(
        self,
        sheet_to_convert: Union[str, int, None] = None,  # Relevant for Excel/ODS
        orient: Literal['dict', 'list', 'series', 'split', 'records', 'index'] = 'records', # Corrected type hint
        date_format: Union[str, None] = 'iso',  # 'iso', 'epoch', or None
        csv_encoding: str = 'utf-8', # For reading CSV files
        excel_ods_engine: PandasReadEngineType = None # For Excel/ODS, e.g., 'openpyxl', 'xlrd', 'odf'
        ) -> bool:
        """
        Converts an Excel, ODS, or CSV file (or a specific Excel/ODS sheet)
        into an equivalent JSON format.

        Args:
            sheet_to_convert (str | int | None, optional):
                - For Excel/ODS:
                    - If None (default): Converts all sheets. The JSON output will be a
                      dictionary where keys are sheet names and values are the JSON
                      representation of each sheet.
                    - If str: Name of the specific sheet to convert.
                    - If int: Index of the specific sheet to convert (0-based).
                    If a specific sheet is requested, the JSON output will directly be
                    the representation of that sheet.
                - For CSV: This parameter is ignored. The entire CSV is processed.
            orient (str, optional): Pandas DataFrame.to_dict() orientation for each sheet/CSV.
                Default: 'records'. See pandas.DataFrame.to_dict() documentation.
            date_format (str | None, optional): Format for datetime objects.
                - 'iso' (default): ISO8601 format (e.g., '2023-10-27T10:30:00').
                - 'epoch': Milliseconds since epoch.
                - None: Pandas default (often Timestamps). 'iso' is generally safer for JSON.
            csv_encoding (str, optional): Encoding for reading CSV files. Default is 'utf-8'.
            excel_ods_engine (str | None, optional): Pandas engine for reading Excel or ODS files.
                - For Excel: 'openpyxl' (for .xlsx), 'xlrd' (for .xls).
                - For ODS: 'odf' (requires 'odfpy' library).
                If None, pandas auto-detects based on file extension and installed libraries.

        Returns:
            bool: True if conversion was successful, False otherwise.
        """

        input_filepath = self.input_filepath
        json_filepath = self.json_filepath

        try:

            if not input_filepath.exists():
                print(f"Error: Input file not found at {input_filepath}")
                return False

            # Ensure output directory exists
            json_filepath.parent.mkdir(parents=True, exist_ok=True)

            file_suffix = input_filepath.suffix.lower()
            output_data_final: Union[Dict[str, Any], List[Dict[str, Any]]] = {}

            dataframes_to_process: list[tuple[pd.DataFrame, str | None]] = []

            current_engine: PandasReadEngineType = excel_ods_engine

            if file_suffix == '.csv':
                if sheet_to_convert is not None:
                    print(f"Info: 'sheet_to_convert' parameter ('{sheet_to_convert}') is ignored for CSV file '{input_filepath.name}'. Processing entire CSV.")
                try:
                    df = pd.read_csv(input_filepath, encoding=csv_encoding)
                    dataframes_to_process.append((df, None))
                except Exception as e:
                    print(f"Error reading CSV file '{input_filepath.name}': {e}")
                    return False

            elif file_suffix in ['.xls', '.xlsx', '.ods']:
                try:
                    if file_suffix == '.ods':
                        if current_engine is None:
                            current_engine = 'odf'
                        elif current_engine != 'odf':
                            print(f"Warning: Specified engine '{current_engine}' may not be optimal for ODS. Forcing 'odf'.")
                            current_engine = 'odf'

                    if sheet_to_convert is not None:
                        df = pd.read_excel(input_filepath, sheet_name=sheet_to_convert, engine=current_engine)
                        dataframes_to_process.append((df, None))

                    else:
                        excel_file = pd.ExcelFile(input_filepath, engine=current_engine)
                        if not excel_file.sheet_names:
                            print(f"Warning: File '{input_filepath.name}' contains no sheets.")
                        for sheet_name in excel_file.sheet_names:
                            df = excel_file.parse(sheet_name) # engine is inherited
                            dataframes_to_process.append((df, sheet_name))
                except ImportError as ie:
                    if 'odfpy' in str(ie).lower() and file_suffix == '.ods':
                        print(f"Error reading ODS file '{input_filepath.name}': The 'odfpy' library is required. Please install it using 'pip install odfpy'.")
                    elif 'xlrd' in str(ie).lower() and file_suffix == '.xls':
                        print(f"Error reading .xls file '{input_filepath.name}': The 'xlrd' library might be required. Please install it using 'pip install xlrd'.")
                    elif 'openpyxl' in str(ie).lower() and file_suffix == '.xlsx':
                        print(f"Error reading .xlsx file '{input_filepath.name}': The 'openpyxl' library might be required. Please install it using 'pip install openpyxl'.")
                    else:
                        print(f"ImportError reading file '{input_filepath.name}': {ie}")
                    return False
                except Exception as e:
                    print(f"Error reading Excel/ODS file '{input_filepath.name}': {e}")
                    return False
            else:
                print(f"Error: Unsupported file type: '{file_suffix}'. Please provide a CSV, XLS, XLSX, or ODS file.")
                return False

            if not dataframes_to_process and file_suffix in ['.xls', '.xlsx', '.ods'] and sheet_to_convert is None:
                 print(f"Info: No dataframes were loaded from '{input_filepath.name}'. Output JSON will be empty if processing all sheets from an empty file.")
            elif not dataframes_to_process and not (file_suffix in ['.xls', '.xlsx', '.ods'] and sheet_to_convert is None):
                 pass

            is_direct_output = len(dataframes_to_process) == 1 and dataframes_to_process[0][1] is None
            temp_processed_data: Dict[str, Any] = {}

            for df_original, name_key in dataframes_to_process:
                df = df_original.copy()

                # Handle datetime columns with improved detection and conversion
                if date_format:
                    # Check for datetime columns using multiple approaches
                    datetime_columns = []

                    # Method 1: Use pandas dtype detection
                    datetime_columns.extend(df.select_dtypes(include=['datetime64[ns]', 'datetime', 'datetimetz']).columns.tolist())

                    # Method 2: Check for datetime objects in each column
                    for col in df.columns:
                        if col not in datetime_columns:
                            # Sample a few non-null values to check type
                            sample_values = df[col].dropna().head(10)
                            if len(sample_values) > 0:
                                for val in sample_values:
                                    if isinstance(val, (datetime, date, pd.Timestamp)):
                                        datetime_columns.append(col)
                                        break

                    # Convert datetime columns
                    for col_name in datetime_columns:
                        try:
                            if date_format == 'iso':
                                df[col_name] = df[col_name].apply(lambda x: self._convert_to_iso(x))
                            elif date_format == 'epoch':
                                df[col_name] = df[col_name].apply(lambda x: self._convert_to_epoch(x))
                        except Exception as e_date:
                            print(f"Warning: Could not fully convert date column '{col_name}' in '{name_key or input_filepath.name}' using format '{date_format}'. Error: {e_date}")

                # Replace NaN values with None for JSON compatibility
                df = df.astype(object).where(pd.notnull(df), None)

                # Final safety check: convert any remaining datetime objects
                for col in df.columns:
                    df[col] = df[col].apply(lambda x: self._safe_datetime_convert(x, date_format))

                current_json_segment = df.to_dict(orient=orient)

                if is_direct_output:
                    output_data_final = current_json_segment
                    break
                else:
                    if name_key is not None:
                        temp_processed_data[name_key] = current_json_segment

            if not is_direct_output:
                output_data_final = temp_processed_data

            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data_final, f, indent=4, ensure_ascii=False)

            print(f"Successfully converted '{input_filepath.name}' to '{json_filepath.name}'")
            return True

        except FileNotFoundError:
            print(f"Error: Input file not found at {input_filepath.name}")
            return False
        except ValueError as ve:
            print(f"ValueError during conversion of '{input_filepath.name}': {ve}")
            return False
        except Exception as e:
            print(f"An unexpected error occurred during conversion of '{input_filepath.name}': {e}")
            return False

    def _convert_to_iso(self, value):
        """Convert datetime-like objects to ISO format string."""
        if pd.isnull(value) or value is None:
            return None

        try:
            if isinstance(value, str):
                return value  # Already a string
            elif hasattr(value, 'isoformat'):
                return value.isoformat()
            elif isinstance(value, pd.Timestamp):
                return value.isoformat()
            else:
                return str(value)
        except:
            return str(value) if value is not None else None

    def _convert_to_epoch(self, value):
        """Convert datetime-like objects to epoch milliseconds."""
        if pd.isnull(value) or value is None:
            return None

        try:
            if isinstance(value, (int, float)):
                return int(value)  # Assume already epoch
            elif hasattr(value, 'timestamp'):
                return int(value.timestamp() * 1000)
            elif isinstance(value, pd.Timestamp):
                return int(value.timestamp() * 1000)
            else:
                return str(value)
        except:
            return str(value) if value is not None else None

    def _safe_datetime_convert(self, value, date_format):
        """Final safety conversion for any remaining datetime objects."""
        if pd.isnull(value) or value is None:
            return None

        # If it's a datetime-like object, convert it
        if isinstance(value, (datetime, date, pd.Timestamp)):
            if date_format == 'iso':
                return self._convert_to_iso(value)
            elif date_format == 'epoch':
                return self._convert_to_epoch(value)
            else:
                return str(value)

        return value


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
        print("Test 1: scaned pdf page, bytes")
        with open("/home/cyto/Documents/scanned.pdf", "rb") as imgpdf:
            conv= Convertor(file_bytes= imgpdf.read(), suffix= ".pdf", file_type= "pdf")
            print(conv.output)

        # print("Test 2: JD pdf, bytes")
        # with open("/home/cyto/dev/pembotdir/jds/PM Trainee.pdf", "rb") as imgpdf:
        #     conv= Convertor(file_bytes= imgpdf.read(), suffix= ".pdf", file_type= "pdf")
        #     print(conv.output)

        # print("Test 3: excel schedule, bytes")
        # with open("/home/cyto/Downloads/Assignment schedule.xlsx", "rb") as imgpdf:
        #     conv= Convertor(file_bytes= imgpdf.read(), suffix= ".xlsx", file_type= "excel")
        #     print(conv.output)

        # without bytes example:
        print("Test 4: scanned pdf, path")
        conv= Convertor(myfile= Path('/home/cyto/Documents/scanned.pdf'), output_dir= Path('/home/cyto/Documents'))
        print(conv.output)

        # print("Test 5: schedule excel, path")
        # conv= Convertor(myfile= Path('/home/cyto/Downloads/Assignment schedule.xlsx'), output_dir= Path('/home/cyto/Downloads'))
        # print(conv.output)
    except FileNotFoundError as fe:
        print("file not found, modify the driver code to get sample files to test:\n\n", fe)
    except Exception as e:
        print("unhandled: ", e)

    print("Test Run End.")
