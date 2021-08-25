import numpy as np
import tempfile
from binascii import hexlify, unhexlify

# Compressed numpy array
# Convert compressed numpy array into string for database storage (approximately 3-4 times shorter than numpy.ndarray.tobytes)
# Maybe improve by compressing the string in an extra step: https://www.pythonpool.com/string-compression-python/


def get_string_from_siamese_vector(siamese_vector):
    file_siamese_path = 'temp_siamese.npz'

    np.savez_compressed(file_siamese_path, siamese_vector)

    with open(file_siamese_path, 'rb') as file:
        data = file.read()
    os.remove(file_siamese_path)
    template_data_str = hexlify(data).decode('latin1')
    
    return template_data_str

def get_siamese_vector_from_string(template_data_str_back):
    data_back = unhexlify(template_data_str_back.encode('latin1'))

    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(data_back)
        output = np.load(path)['arr_0']
    finally:
        os.remove(path)

    return output
