import utils.storage as storage
from PIL import Image
import numpy as np
import io as io


container = storage.get_storage_container('expency-input')
provider = storage.get_storage_provider()

obj = container.get_object('IMG_20221010_152054_crop.jpg')
np_image = np.frombuffer(b''.join(obj.as_stream(storage.buffer_size)), dtype=np.uint8)
input_img = Image.open(io.BytesIO(np_image))

input_img.save('.test.png')