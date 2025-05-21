import mimetypes
# import pdfplumber
from pembot.pdf2markdown.extract import MarkdownPDFExtractor


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
    def __init__(self, myfile: str):
        mt= mimetypes.guess_file_type(myfile)[0]
        if mt == 'application/json':
            print("the file was json")
        elif mt == 'application/pdf':
            print("the file was pdf")
            extractor= MarkdownPDFExtractor(myfile)
            print(extractor.extract())


        elif mt == 'text/csv':
            print("the file was csv")
        elif mt in EXCEL_FILE_TYPES:
            print("the file was excel spreadsheet")
        elif mt == 'application/vnd.oasis.opendocument.spreadsheet':
            print("the file was opentype spreadsheet")
        else:
            print(mt)



if __name__ == '__main__':
    conv= Convertor('/home/cyto/Documents/jds/hcltech_ai_engg.pdf')

