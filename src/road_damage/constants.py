"""Project-wide dataset constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLASS_NAMES = ("D00", "D10", "D20", "D40")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
CLASS_DESCRIPTIONS = {
    "D00": "longitudinal crack",
    "D10": "transverse crack",
    "D20": "alligator crack",
    "D40": "pothole",
}

COUNTRIES = (
    "China_Drone",
    "China_MotorBike",
    "Czech",
    "India",
    "Japan",
    "Norway",
    "United_States",
)
DEFAULT_COUNTRIES = ("India",)

FIGSHARE_ARTICLE_ID = 21431547
FIGSHARE_FILE_ID = 38030910
FIGSHARE_DOWNLOAD_URL = f"https://ndownloader.figshare.com/files/{FIGSHARE_FILE_ID}"
FIGSHARE_MD5 = "b62bd51d2ffcfaa76c60f234f0cc2bb3"
FIGSHARE_SIZE_BYTES = 13_264_172_619
ARCHIVE_NAME = "RDD2022_released_through_CRDDC2022.zip"
