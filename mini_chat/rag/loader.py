
import os
from typing import List, Tuple


def load_documents_from_folder(folder_path: str = "data") -> List[Tuple[str, str]]:
    """Load .md/.txt files from a folder.

    Returns:
        A list of (source_name, text).

    Notes:
        We sort filenames to make the build deterministic.
    """
    documents: List[Tuple[str, str]] = []
    if not os.path.isdir(folder_path):
        return documents

    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if not (filename.endswith(".txt") or filename.endswith(".md")):
            continue
        if not os.path.isfile(file_path):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append((filename, text))

    return documents
