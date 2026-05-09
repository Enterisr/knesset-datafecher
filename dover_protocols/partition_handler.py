from .logger_config import get_logger


MAX_FILES_FOR_FOLDER = 1000
logger = get_logger(__name__)


class PartitionHandler:
    def __init__(self, max_files_per_folder: int = MAX_FILES_FOR_FOLDER):
        self.max_files_per_folder = max_files_per_folder
        self.current_file_idx = 0

    def get_folder(self) -> str:
        folder_idx = self.current_file_idx // self.max_files_per_folder
        self.current_file_idx += 1
        return f"part_{folder_idx}"
