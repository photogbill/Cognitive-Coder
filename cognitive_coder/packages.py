# SPDX-License-Identifier: Apache-2.0
"""Third-party Python packages a build may need: which are known, which a
request names, and what pip calls them.

Two uses, both about not stopping the build to ask a person:

  * **Install before the first file.** Oct 1, 2026: the spec said
    "using pygame" on its first line; the engine found out at minute 13,
    when `render.py` failed to import it, and the "Install pygame?" dialog
    waited another thirteen minutes for someone to come back to the
    keyboard. A request's packages can be read before any code is written.
  * **Decide without asking.** The host asks before `pip install` because
    the name usually comes from an import line a model wrote, and a typo
    installs whatever is published under the typo. A name on this list,
    or a name the operator's own request mentions, is not that case.

The list is data, not a model of PyPI: add to it freely. Import name →
pip name, identical where they are the same.
"""

from __future__ import annotations

import re

#: import name → pip name. Only well-known, well-maintained packages whose
#: names are not plausible typos of each other.
KNOWN: dict[str, str] = {
    # games, graphics, GUI
    "pygame": "pygame", "pyglet": "pyglet", "arcade": "arcade",
    "pyray": "raylib", "raylib": "raylib", "panda3d": "panda3d",
    "ursina": "ursina", "PySide6": "PySide6", "PyQt6": "PyQt6",
    "PyQt5": "PyQt5", "wx": "wxPython", "kivy": "kivy",
    "dearpygui": "dearpygui", "customtkinter": "customtkinter",
    "pygame_gui": "pygame_gui",
    # numbers, data, plots
    "numpy": "numpy", "scipy": "scipy", "pandas": "pandas",
    "matplotlib": "matplotlib", "sympy": "sympy", "numba": "numba",
    "sklearn": "scikit-learn", "networkx": "networkx", "shapely": "shapely",
    "polars": "polars", "pyarrow": "pyarrow", "openpyxl": "openpyxl",
    "xlsxwriter": "XlsxWriter", "h5py": "h5py", "plotly": "plotly",
    "seaborn": "seaborn", "statsmodels": "statsmodels",
    # images, audio, video
    "PIL": "pillow", "pillow": "pillow", "cv2": "opencv-python",
    "imageio": "imageio", "skimage": "scikit-image", "sounddevice":
    "sounddevice", "soundfile": "soundfile", "pyaudio": "PyAudio",
    "pydub": "pydub", "moviepy": "moviepy",
    # web, network, files
    "requests": "requests", "httpx": "httpx", "aiohttp": "aiohttp",
    "flask": "flask", "fastapi": "fastapi", "uvicorn": "uvicorn",
    "django": "django", "websockets": "websockets", "bs4":
    "beautifulsoup4", "lxml": "lxml", "yaml": "pyyaml", "toml": "toml",
    "tomli": "tomli", "dotenv": "python-dotenv", "paramiko": "paramiko",
    "serial": "pyserial", "docx": "python-docx", "fitz": "PyMuPDF",
    "pypdf": "pypdf", "reportlab": "reportlab", "markdown": "markdown",
    "jinja2": "jinja2", "rich": "rich", "click": "click", "typer": "typer",
    "tqdm": "tqdm", "colorama": "colorama", "pydantic": "pydantic",
    "attrs": "attrs", "dateutil": "python-dateutil", "pytz": "pytz",
    "cryptography": "cryptography", "Crypto": "pycryptodome",
    "psutil": "psutil", "watchdog": "watchdog", "sqlalchemy": "SQLAlchemy",
    # testing and tooling
    "pytest": "pytest", "hypothesis": "hypothesis", "mypy": "mypy",
    "ruff": "ruff", "black": "black", "coverage": "coverage",
}

#: Standard-library and builtin names that look like packages in prose
#: ("use json", "the math module") and must never be installed.
STDLIB: frozenset = frozenset({
    "json", "math", "random", "time", "datetime", "os", "sys", "re",
    "pathlib", "typing", "dataclasses", "collections", "itertools",
    "functools", "unittest", "logging", "argparse", "subprocess", "sqlite3",
    "csv", "io", "abc", "enum", "struct", "socket", "threading", "asyncio",
    "tkinter", "turtle", "curses", "wave", "array", "copy", "heapq",
    "bisect", "statistics", "decimal", "fractions", "string", "textwrap",
    "shutil", "tempfile", "glob", "pickle", "shelve", "hashlib", "hmac",
    "secrets", "uuid", "queue", "select", "signal", "http", "urllib",
    "email", "html", "xml", "zipfile", "tarfile", "gzip", "base64",
    "binascii", "operator", "contextlib", "inspect", "traceback",
    "warnings", "weakref", "types", "numbers", "cmath", "pprint",
    "configparser", "getpass", "platform", "ctypes", "multiprocessing",
    "concurrent", "venv", "pip", "setuptools", "wheel",
})


#: KNOWN names that are also ordinary words. A request saying "click the
#: button", "rich colours" or "arcade-style" names no package; these are
#: never read out of prose, only approved when a model actually imports
#: them.
AMBIGUOUS: frozenset = frozenset({
    "arcade", "rich", "click", "typer", "attrs", "markdown", "coverage",
    "serial", "docx", "toml", "yaml", "wx", "fitz", "black", "ruff",
    "hypothesis", "requests", "websockets", "watchdog", "statistics",
})


def pip_name(import_name: str) -> str:
    """What pip calls it; the import name itself when not listed."""
    return KNOWN.get(import_name, import_name)


def is_known(name: str) -> bool:
    """A package the host may install without asking a person first."""
    n = str(name or "").strip()
    if not n:
        return False
    if n in KNOWN:
        return True
    low = n.lower()
    return any(low == v.lower() for v in KNOWN.values())


def mentioned_in(text: str, name: str) -> bool:
    """Does the operator's own request name this package (import or pip
    name), as a whole word, in any case?"""
    n = str(name or "").strip()
    if not n or not text:
        return False
    names = {n, pip_name(n)}
    for k, v in KNOWN.items():
        if v.lower() == n.lower():
            names.add(k)
    for candidate in names:
        if re.search(rf"(?<![\w.-]){re.escape(candidate)}(?!\w)", text,
                     flags=re.I):
            return True
    return False


def packages_in(text: str) -> list[tuple[str, str]]:
    """``[(import_name, pip_name)]`` for every KNOWN package the request
    names, in first-mention order. Standard-library names never appear
    (they are not in KNOWN), nor do KNOWN names that are ordinary words
    (`AMBIGUOUS`): "a pygame racing game" names pygame; "an arcade-style
    racer" names nothing."""
    found: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for imp, pip in KNOWN.items():
        if pip in seen or imp in AMBIGUOUS or pip in AMBIGUOUS:
            continue
        for candidate in {imp, pip}:
            # "pygame-based" names pygame; "scikit-learn" does not name a
            # package called "learn" — hence the guard before, none after.
            m = re.search(rf"(?<![\w.-]){re.escape(candidate)}(?!\w)",
                          text or "", flags=re.I)
            if m:
                found.append((m.start(), imp, pip))
                seen.add(pip)
                break
    found.sort()
    return [(imp, pip) for _pos, imp, pip in found]
