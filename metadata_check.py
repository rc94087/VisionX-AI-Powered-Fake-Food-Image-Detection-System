
from PIL import Image
from PIL.ExifTags import TAGS

def check_metadata(image_path):
    img = Image.open(image_path)
    exif = img._getexif()

    if exif is None:
        return "No metadata found"

    data = {}
    for tag_id, value in exif.items():
        tag = TAGS.get(tag_id, tag_id)
        data[tag] = value

    suspicious = ["Software", "Generator", "AI", "Stable Diffusion"]
    for key in data:
        if any(s.lower() in str(data[key]).lower() for s in suspicious):
            return "AI-related metadata detected"

    return "Metadata appears normal"
