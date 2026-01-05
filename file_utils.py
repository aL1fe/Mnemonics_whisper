import os


async def save_file(file, upload_folder) -> str | None:
    try:
        os.makedirs(upload_folder, exist_ok=True)  # Create folder if it does not exist
        file_path = os.path.join(upload_folder, file.filename)  # Form the full path for the file

        with open(file_path, "wb") as f:
            f.write(await file.read())

        return file_path
    except OSError as e:
        print(f"Error saving file {file.filename}: {e}")


async def delete_file(file, upload_folder) -> None:
    file_path = os.path.join(upload_folder, file.filename)
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
    except OSError as e:
        print(f"Error deleting file {file_path}: {e}")

