from pathlib import Path
import argparse


# Directories that are never shown in any mode.
# These are development/runtime directories rather than project content.
ALWAYS_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "ENV",
    "deeplearning_env",
    "__pycache__",
    ".ipynb_checkpoints",
}


# Files that are never shown in any mode.
ALWAYS_EXCLUDED_FILES = {
    ".DS_Store",
    "Thumbs.db",
}


# Files shown in Project Tree mode.
PROJECT_EXTENSIONS = {
    ".ipynb",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".py",
    ".md",
}


# Important project files without a conventional extension.
PROJECT_FILES = {
    ".gitignore",
    "LICENSE",
}


# Additional source/data files shown in Detailed Tree mode.
DETAILED_EXTENSIONS = {
    ".tex",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".bib",
    ".sty",
    ".toml",
    ".xml",
    ".html",
    ".css",
}


# Generated/cache files hidden from Project and Detailed Tree modes.
# Full Tree still shows these.
GENERATED_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".log",
    ".aux",
    ".out",
    ".toc",
    ".lof",
    ".lot",
    ".fls",
    ".fdb_latexmk",
    ".synctex",
    ".synctex.gz",
}


def should_exclude(path: Path, mode: str) -> bool:
    """Return True if the path should be excluded for the selected mode."""
    if path.is_dir():
        return path.name in ALWAYS_EXCLUDED_DIRS

    if path.name in ALWAYS_EXCLUDED_FILES:
        return True

    if mode == "full":
        return False

    extension = path.suffix.lower()

    if extension in GENERATED_EXTENSIONS:
        return True

    if mode == "project":
        return (
            path.name not in PROJECT_FILES
            and extension not in PROJECT_EXTENSIONS
        )

    if mode == "detailed":
        return (
            path.name not in PROJECT_FILES
            and extension not in PROJECT_EXTENSIONS
            and extension not in DETAILED_EXTENSIONS
        )

    return False


def print_tree(
    directory: Path,
    mode: str,
    prefix: str = "",
) -> None:
    """Recursively print the directory tree."""
    entries = sorted(
        [
            entry
            for entry in directory.iterdir()
            if not should_exclude(entry, mode)
        ],
        key=lambda path: (path.is_file(), path.name.lower()),
    )

    for index, entry in enumerate(entries):
        is_last = index == len(entries) - 1

        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{entry.name}")

        if entry.is_dir():
            extension = "    " if is_last else "│   "
            print_tree(
                entry,
                mode,
                prefix + extension,
            )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Print the COMP395 project directory tree."
    )

    parser.add_argument(
        "--mode",
        choices=["project", "detailed", "full"],
        default="project",
        help="Tree detail level.",
    )

    return parser.parse_args()


def main() -> None:
    """Print the COMP395 project tree."""
    args = parse_arguments()

    project_root = Path(__file__).resolve().parent.parent

    print(f"{project_root.name}/")
    print_tree(
        project_root,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()