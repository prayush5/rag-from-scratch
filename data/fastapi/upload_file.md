# Handling File Uploads in FastAPI

FastAPI provides two ways to accept file uploads: `File` for raw bytes and `UploadFile` for streaming large files directly.

## UploadFile Advantages

- Uses a Python `SpooledTemporaryFile` to store memory efficiently up to a threshold.
- Suitable for large files like images, videos, or PDFs without consuming all system memory.
- Provides async methods such as `.read()`, `.write()`, and `.close()`.
