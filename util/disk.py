import gzip
from diskcache import FanoutCache, Disk
from io import BytesIO
from util.logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class GzipDisk(Disk):
    def store(self, value, read, key=None):
        """
        Stores a value in the cache, compressing it with gzip if it's binary data.

        Args:
            value: The value to be stored.
            read: A flag indicating whether the value should be read from a file-like object.
            key: The key associated with the value (unused in this method).

        Returns:
            The result of the store operation from the superclass.
        """
        if isinstance(value, bytes):
            if read:
                value = value.read()
                read = False

            str_io = BytesIO()
            with gzip.GzipFile(mode='wb', compresslevel=1, fileobj=str_io) as gz_file:
                for offset in range(0, len(value), 2**30):
                    gz_file.write(value[offset:offset + 2**30])

            value = str_io.getvalue()

        return super().store(value, read)

    def fetch(self, mode, filename, value, read):
        """
        Fetches a value from the cache, decompressing it with gzip if it's binary data.

        Args:
            mode: The mode of the fetch operation (e.g., MODE_BINARY).
            filename: The filename associated with the value (unused in this method).
            value: The value to be fetched.
            read: A flag indicating whether the value should be read into a file-like object.

        Returns:
            The fetched value, possibly decompressed.
        """
        value = super().fetch(mode, filename, value, read)

        if mode == MODE_BINARY:
            with gzip.GzipFile(mode='rb', fileobj=BytesIO(value)) as gz_file:
                read_csio = BytesIO()

                while True:
                    uncompressed_data = gz_file.read(2**30)
                    if uncompressed_data:
                        read_csio.write(uncompressed_data)
                    else:
                        break

                value = read_csio.getvalue()

        return value

def get_cache(scope_str):
    """
    Creates a FanoutCache instance with a custom GzipDisk storage backend.

    Args:
        scope_str: A string representing the scope of the cache.

    Returns:
        A FanoutCache instance.
    """
    return FanoutCache('data-unversioned/cache/' + scope_str,
                       disk=GzipDisk,
                       shards=64,
                       timeout=1,
                       size_limit=3e11)
