from dataclasses import dataclass
from typing import Any


@dataclass
class FileWhitelistData:
    """Data about a file that has been read and can be modified."""

    file_hash: str
    # List of line ranges that have been read (inclusive start, inclusive end)
    # E.g., [(1, 10), (20, 30)] means lines 1-10 and 20-30 have been read
    line_ranges_read: list[tuple[int, int]]
    # Total number of lines in the file
    total_lines: int

    def get_percentage_read(self) -> float:
        """Calculate percentage of file read based on line ranges."""
        if self.total_lines == 0:
            return 100.0

        # Count unique lines read
        lines_read: set[int] = set()
        for start, end in self.line_ranges_read:
            lines_read.update(range(start, end + 1))

        return (len(lines_read) / self.total_lines) * 100.0

    def is_read_enough(self) -> bool:
        """Check if enough of the file has been read (>=99%)"""
        return self.get_percentage_read() >= 99

    def get_unread_ranges(self) -> list[tuple[int, int]]:
        """Return a list of line ranges (start, end) that haven't been read yet.

        Returns line ranges as tuples of (start_line, end_line) in 1-indexed format.
        If the whole file has been read, returns an empty list.
        """
        if self.total_lines == 0:
            return []

        # First collect all lines that have been read
        lines_read: set[int] = set()
        for start, end in self.line_ranges_read:
            lines_read.update(range(start, end + 1))

        # Generate unread ranges from the gaps
        unread_ranges: list[tuple[int, int]] = []
        start_range = None

        for i in range(1, self.total_lines + 1):
            if i not in lines_read:
                if start_range is None:
                    start_range = i
            elif start_range is not None:
                # End of an unread range
                unread_ranges.append((start_range, i - 1))
                start_range = None

        # Don't forget the last range if it extends to the end of the file
        if start_range is not None:
            unread_ranges.append((start_range, self.total_lines))

        return unread_ranges

    def add_range(self, start: int, end: int) -> None:
        """Add a new range of lines that have been read."""
        self.line_ranges_read.append((start, end))

    def serialize(self) -> dict[str, Any]:
        """Convert to a serializable dictionary."""
        return {
            "file_hash": self.file_hash,
            "line_ranges_read": self.line_ranges_read,
            "total_lines": self.total_lines,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "FileWhitelistData":
        """Create from a serialized dictionary."""
        return cls(
            file_hash=data.get("file_hash", ""),
            line_ranges_read=data.get("line_ranges_read", []),
            total_lines=data.get("total_lines", 0),
        )
