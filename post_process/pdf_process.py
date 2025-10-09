# given a folder and argument, this function will add all the selected pdfs in the folder to a single pdf

import os
from PyPDF2 import PdfFileMerger


def merge_pdfs(folder_path, output_path, containKeywords=None):
    # Get all the files in the folder
    files = os.listdir(folder_path)

    # Filter files based on the keywords
    if containKeywords:
        files = [f for f in files if any(k in f for k in containKeywords)]

    # Create a PdfFileMerger object
    merger = PdfFileMerger()

    # Iterate over the files and append them to the merger object
    for file in files:
        file_path = os.path.join(folder_path, file)
        merger.append(file_path)

    # Write the merged PDF to the output path
    merger.write(output_path)
    merger.close()