# SPDX-License-Identifier: Apache-2.0
"""The static screen (§6.3) — a positive and a negative for every language.

Read `guard.py`'s docstring first: this is a screen against ACCIDENTS, not a
security boundary. These tests hold it to that job in both directions:

  * it must catch the obvious accident in every language it runs — the
    review found `os.remove('/etc/hosts')` in project mode, Rust's
    `use std::process::{Command}` brace form, Go's `net.Dial`, `find
    -delete`, `Remove-Item -Recurse` and `rd /s` all passing, and Ruby,
    Lua, Zig and SQL with no rules at all;
  * it must NOT refuse code for what its comments and docstrings SAY. It
    blocked `# never run rm -rf here` and a docstring reading "never uses
    requests", and with `re.I` on every rule a C function named `System`
    was "shell execution". A screen that cries wolf teaches the model to
    rewrite correct code.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import guard, langs  # noqa: E402


def _blocked(code, lang, project=False):
    return guard.blocked(guard.scan(code, lang, project))


# (lang, code, project_mode) — each must be REFUSED.
POSITIVE = [
    ("python", "import os\nos.remove('/etc/hosts')\n", True),
    ("python", "from pathlib import Path\nPath('../../x').unlink()\n", True),
    ("python", "import shutil\nshutil.move('src', '../../elsewhere')\n",
     True),
    ("python", "open('/home/me/.bashrc', 'w').write('x')\n", True),
    ("python", "from os import system\nsystem('ls')\n", False),
    ("python", "import aiohttp\n", False),
    ("python", "import importlib\nm = importlib.import_module('x')\n",
     False),
    ("python", "import os\nopen(os.path.expanduser('~/.bashrc'), 'w')\n",
     True),
    ("bash", "find . -name '*.o' -delete\n", False),
    ("bash", "dd if=/dev/zero of=/dev/sda bs=1M\n", False),
    ("bash", "rm --recursive --force build\n", False),
    ("bash", "ssh user@host 'ls'\n", False),
    ("bash", "echo x > /etc/passwd\n", True),
    ("c", '#include <stdio.h>\nint main(){ remove("/etc/hosts"); }\n', True),
    ("c", 'int main(){ CreateProcessA(0, "cmd", 0); }\n', False),
    ("cpp", 'std::filesystem::remove_all("build");\n', True),
    ("rust", "use std::process::{Command, Stdio};\n", False),
    ("rust", "use std::net::{TcpStream};\n", False),
    ("rust", "fn f<'a>(x: &'a str) -> &'a str { x }\n"
             "fn main() { std::process::Command::new(\"sh\"); }\n", False),
    ("go", 'package main\nimport "os"\nfunc main() { os.Remove("/etc/hosts")'
           ' }\n', True),
    ("go", 'package main\nimport n "net"\nfunc main() { n.Dial("tcp", "x") }'
           '\n', False),
    ("javascript", "fs.rmSync(dir, {\n  recursive: true\n});\n", False),
    ("javascript", "import { exec } from 'node:child_process';\n", False),
    ("javascript", "import axios from 'axios';\n", False),
    ("javascript", "require('vm').runInNewContext('x');\n", False),
    ("java", "Process p = new ProcessBuilder(\"ls\").start();\n", False),
    ("csharp", 'Directory.Delete("/", true);\n', True),
    ("powershell", "Remove-Item -Recurse -Force C:\\Users\n", True),
    ("powershell", "remove-item -recurse build\n", False),
    ("powershell", "Start-Process notepad\n", False),
    ("batch", "rd /s /q C:\\Users\n", True),
    ("batch", "RD /S /Q build\n", False),
    ("batch", "del /s /q *.*\n", False),
    ("ruby", "system('ls')\n", False),
    ("ruby", "FileUtils.rm_rf('/')\n", False),
    ("ruby", "require 'net/http'\nNet::HTTP.get(u)\n", False),
    ("ruby", "out = `ls`\n", False),
    ("lua", "os.execute('rm -r x')\n", False),
    ("lua", "local s = require('socket')\n", False),
    ("zig", "const c = std.process.Child.init(argv, a);\n", False),
    ("zig", "try std.fs.cwd().deleteTree(\"build\");\n", False),
    ("sql", "ATTACH DATABASE '/etc/passwd' AS x;\n", False),
    ("sql", "SELECT load_extension('evil');\n", False),
    ("gdscript", 'OS.shell_open("http://example.com")\n', False),
]

# (lang, code) — each must PASS: the dangerous words are only in comments,
# docstrings or strings, or the name merely resembles one.
NEGATIVE = [
    ("python", '"""This module never uses requests or subprocess."""\n'
               "x = 1\n"),
    ("python", "# kill -9 the old one first? no.\nx = 1\n"),
    ("python", "s = 'format d: done'\nsocket_count = 0\n"),
    ("python", "from urllib.parse import quote\n"),
    ("python", "def evaluate(x):\n    return x\n\nprint(evaluate(1))\n"),
    ("python", "app.exec()\n"),
    ("bash", "# never run rm -rf here\necho ok\n"),
    ("bash", "make build > /dev/null 2>&1\necho $# ${#arr[@]}\n"),
    ("c", "/* later we connect( to nothing */\nint main(){ return 0; }\n"),
    ("c", "int System(void) { return 0; }\nint main(){ return System(); }\n"),
    ("cpp", '// std::system("rm -rf /") is what NOT to do\nint main(){}\n'),
    ("rust", "// Command::new is not used here\nfn main() {}\n"),
    ("rust", "fn f<'a>(x: &'a str) -> &'a str { x }\nfn main() {}\n"),
    ("go", "package main\n// net.Dial is not used\nfunc main() {}\n"),
    ("javascript", "// unlike child_process, this is pure\nconst x = 1;\n"),
    ("javascript", 'const msg = "call fetch( later";\n'),
    ("java", "// ProcessBuilder is not used\nclass A {}\n"),
    ("powershell", "# Remove-Item -Recurse is dangerous\nWrite-Host ok\n"),
    ("batch", "REM rd /s is dangerous\necho ok\n"),
    ("batch", ":: del /s is dangerous\necho ok\n"),
    ("ruby", "# system('ls') would be bad\nputs 1\n"),
    ("lua", "-- os.execute is not used\nprint(1)\n"),
    ("lua", "--[[ io.popen is not used ]]\nprint(1)\n"),
    ("zig", "// std.process.Child is not used\npub fn main() void {}\n"),
    ("sql", "-- ATTACH DATABASE is not used\nSELECT 1;\n"),
    ("gdscript", "# OS.execute is not used\nfunc _ready():\n\tpass\n"),
]


@pytest.mark.parametrize("lang,code,project", POSITIVE,
                         ids=[f"{p[0]}:{p[1][:30]}" for p in POSITIVE])
def test_an_obvious_accident_is_refused(lang, code, project):
    assert _blocked(code, lang, project), f"{lang} let through: {code!r}"


@pytest.mark.parametrize("lang,code", NEGATIVE,
                         ids=[f"{n[0]}:{n[1][:30]}" for n in NEGATIVE])
def test_words_in_comments_and_strings_are_not_code(lang, code):
    found = guard.scan(code, lang, project_mode=True)
    assert not guard.blocked(found), (
        f"{lang} refused {code!r}: {[f.one_line() for f in found]}")


def test_every_language_the_engine_runs_has_rules_of_its_own():
    covered = guard.languages_with_rules()
    missing = sorted(set(langs.ids()) - covered)
    assert not missing, f"no language-specific rules for {missing}"


def test_a_local_file_deletion_is_a_warning_not_a_refusal():
    """Deleting a file the program itself made is ordinary; only one aimed
    outside the project is refused."""
    found = guard.scan("import os\nos.remove(tmp_path)\n", "python", True)
    assert not guard.blocked(found)
    assert "deletes or moves files" in guard.advisory(found)


def test_the_finding_quotes_the_real_source_and_line():
    code = "x = 1\n# fine\nimport os\nos.system('ls')\n"
    hard = [f for f in guard.scan(code, "python") if f.severity == "block"]
    assert hard[0].line == 4 and "os.system" in hard[0].match


def test_the_docstring_still_says_what_this_is_not():
    assert "not a security boundary" in guard.__doc__
