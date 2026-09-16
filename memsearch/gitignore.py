"""A small, pragmatic .gitignore matcher -- not a full spec implementation,
but handles the common cases: comments, blank lines, trailing '/' (dir-only),
leading '/' (root-anchored), '*' and '**' globs, and '!' negation.
"""

from __future__ import annotations

import fnmatch
import posixpath
from pathlib import Path


class GitignoreMatcher:
    def __init__(self, root: Path):
        self.root = root
        self.patterns: list[tuple[str, bool, bool]] = []  # (pattern, dir_only, negate)
        gi = root / ".gitignore"
        if gi.is_file():
            for line in gi.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                negate = line.startswith("!")
                if negate:
                    line = line[1:]
                dir_only = line.endswith("/")
                if dir_only:
                    line = line[:-1]
                anchored = line.startswith("/")
                if anchored:
                    line = line[1:]
                if not anchored:
                    line = f"**/{line}"
                self.patterns.append((line, dir_only, negate))

    def is_ignored(self, path: Path) -> bool:
        if not self.patterns:
            return False
        rel = posixpath.normpath(path.relative_to(self.root).as_posix())
        ignored = False
        for pattern, dir_only, negate in self.patterns:
            if dir_only and not path.is_dir():
                # still allow matching a file inside an ignored dir via the
                # dir pattern below in the directory-skip check; a dir-only
                # pattern shouldn't match a plain file directly.
                if fnmatch.fnmatch(rel, pattern.rstrip("/")):
                    pass
                else:
                    continue
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel, pattern.rstrip("/")):
                ignored = not negate
        return ignored
