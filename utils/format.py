import os 
from dotenv import load_dotenv

load_dotenv()

IMG_MAIN_URL = os.getenv("PRODUCT_PHOTO_ROOT_URL", "")
IMG_FORMAT = os.getenv("PRODUCT_PHOTO_FORMAT", "")

def build_residence_photo(image_path: str | None) -> str:
    if not image_path:
        return ""
    image_path = image_path.replace("photos/", "")
    return IMG_MAIN_URL.rstrip("/") + "/" + image_path.lstrip("/") + IMG_FORMAT