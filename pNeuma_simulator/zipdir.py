import os


def zipdir(path: str, ziph) -> None:
    """
    Zip the directory at the given path.

    Args:
        path (str): The path of the directory to be zipped.
        ziph: The zipfile handle.
    """
    # ziph is zipfile handle
    # https://stackoverflow.com/questions/1855095/
    # https://stackoverflow.com/questions/36740683/
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(").json"):
                os.chdir(root)
                ziph.write(file)
                os.remove(file)
                path_parent = os.path.dirname(os.getcwd())
                os.chdir(path_parent)
