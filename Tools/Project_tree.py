from pathlib import Path
import argparse
import re


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


# Detailed Tree also includes source, data, and configuration files.
DETAILED_EXTENSIONS = PROJECT_EXTENSIONS | {
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


def should_exclude(path: Path, mode: str) -> bool:
    """Return True if the path should be excluded for the selected mode."""
    if path.is_dir():
        return path.name in ALWAYS_EXCLUDED_DIRS or (
            mode == "project" and path.name == ".pytest_cache"
        )

    if path.name in ALWAYS_EXCLUDED_FILES:
        return True

    if mode == "full" or path.name in PROJECT_FILES:
        return False

    extensions = PROJECT_EXTENSIONS if mode == "project" else DETAILED_EXTENSIONS
    return path.suffix.lower() not in extensions


def tree_sort_key(path: Path) -> tuple:
    """List directories first, then sort names naturally (2 before 10)."""
    name = path.name.lower()
    parts = tuple(
        int(part) if part.isdecimal() else part
        for part in re.split(r"(\d+)", name)
    )
    return path.is_file(), parts, name, path.name


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
        key=tree_sort_key,
    )

    for index, entry in enumerate(entries):
        is_last = index == len(entries) - 1

        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{entry.name}")

        if mode == "project" and entry.name == "wandb":
            continue

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
        help=(
            "project (default) shows course deliverables, hides .pytest_cache, "
            "and lists wandb by name only. detailed expands both and adds "
            "source, data, and configuration files. full shows every file "
            "except entries excluded in all modes."
        ),
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
