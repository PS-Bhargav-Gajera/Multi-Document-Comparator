from pathlib import Path
from typing import List, Tuple
from config import MAX_FILE_SIZE_BYTES, MAX_FILES
from utils.logger import get_logger

logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".pdf"}


class InputValidator:
    @staticmethod
    def validate_files(file_paths: List[str]) -> Tuple[List[str], List[str]]:
        valid = []
        errors = []

        if len(file_paths) > MAX_FILES:
            errors.append(f"Maximum {MAX_FILES} files allowed")
            return valid, errors

        for fp in file_paths:
            path = Path(fp)

            if path.suffix.lower() not in ALLOWED_EXTENSIONS:
                errors.append(f"Invalid extension: {path.name}. Only PDF files are allowed.")
                continue

            if not path.exists():
                errors.append(f"File not found: {path.name}")
                continue

            file_size = path.stat().st_size
            if file_size > MAX_FILE_SIZE_BYTES:
                mb = file_size / (1024 * 1024)
                max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
                errors.append(f"File too large: {path.name} ({mb:.1f} MB). Max is {max_mb} MB.")
                continue

            if file_size == 0:
                errors.append(f"Empty file: {path.name}")
                continue

            valid.append(fp)

        return valid, errors
