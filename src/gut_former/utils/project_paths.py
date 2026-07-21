from pathlib import Path


def find_project_root(start_path: str = __file__, marker: str = ".git") -> Path:
    """
    Find the project root by searching upward for a specified file or folder.

    Parameters:
        start_path (str): The path to start the search from. Defaults to the file location of the caller.
        marker (str): The name of the file or directory to look for as the project root marker. Default is ".git".

    Returns:
        Path: The path to the project root directory.
    """
    project_root = Path(start_path).resolve().parent
    while not (project_root / marker).exists() and project_root != project_root.parent:
        project_root = project_root.parent
    return project_root

def find_data_path(*parts: str) -> Path:
    """Path to the project's data directory (optionally joined with subpaths)."""
    return find_project_root() / "data" / Path(*parts)

def find_output_path(*parts: str) -> Path:
    """Path to the project's output directory (optionally joined with subpaths)."""
    return find_project_root() / "output" / Path(*parts)

def find_figures_path(*parts: str) -> Path:
    """Path to the project's figures directory (optionally joined with subpaths)."""
    return find_project_root() / "figures" / Path(*parts)