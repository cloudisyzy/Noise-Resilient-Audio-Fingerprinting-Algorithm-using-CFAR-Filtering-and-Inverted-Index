from .fingerprint import (
    MusicFingerprint,
    MusicFingerprintFromData,
    MusicFingerPrint,            # backward-compatible alias
    MusicFingerPrintFromData,    # backward-compatible alias
)

from .index import (
    inverted_index_table,
    query_from_table,
    music_to_folder_matching,
    folder_to_folder_matching,
    # backward-compatible aliases
    invertedIndexTable,
    queryFromTable,
    musicToFolderMatching,
    folderToFolderMatching,
)

__all__ = [
    "MusicFingerprint",
    "MusicFingerprintFromData",
    "MusicFingerPrint",
    "MusicFingerPrintFromData",
    "inverted_index_table",
    "query_from_table",
    "music_to_folder_matching",
    "folder_to_folder_matching",
    "invertedIndexTable",
    "queryFromTable",
    "musicToFolderMatching",
    "folderToFolderMatching",
]