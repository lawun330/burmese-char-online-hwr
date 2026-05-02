## Myanmar Syllable Handwritten Dataset Browser
## Vibe Coding with ChatGPT by Ye Kyaw Thu, LU Lab., Myanmar
## Last Updated: 23 Mar 2026
## How to run: python dataset_browser.py --help
## E.g.: python dataset_browser.py --dataset dataset --textfile eg-list.txt

import sys
import os
import argparse
import re
from collections import defaultdict

from PyQt5.QtWidgets import (
    QApplication, QWidget, QListWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QDialog, QTextEdit, QMessageBox, QComboBox,
    QLineEdit, QDialogButtonBox,
)
from PyQt5.QtGui import QPainter, QPen, QFont
from PyQt5.QtCore import Qt, QPoint

# -------------------------
# Stroke Viewer Canvas
# -------------------------
class StrokeViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(500, 400)
        self.strokes = []

    def load_file(self, filepath):
        self.strokes = []

        if not os.path.exists(filepath):
            return

        with open(filepath, "r") as f:
            stroke = []
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("STROKE"):
                    if stroke:
                        self.strokes.append(stroke)
                        stroke = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        x, y = int(parts[0]), int(parts[1])
                        stroke.append((x, y))

            if stroke:
                self.strokes.append(stroke)

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(Qt.black, 2)
        painter.setPen(pen)

        for stroke in self.strokes:
            for i in range(1, len(stroke)):
                p1 = QPoint(stroke[i - 1][0], stroke[i - 1][1])
                p2 = QPoint(stroke[i][0], stroke[i][1])
                painter.drawLine(p1, p2)


# -------------------------
# Main Window
# -------------------------
class MainWindow(QWidget):
    def __init__(self, dataset_dir, text_lines):
        super().__init__()

        # Title and dataset directory
        self.setWindowTitle("Handwriting Dataset Browser")
        self.dataset_dir = dataset_dir
        self.text_lines = text_lines

        # Widgets
        self.user_list = QListWidget()
        self.file_list = QListWidget()
        self.char_label = QLabel("Character")
        self.char_label.setAlignment(Qt.AlignCenter)
        self.char_label.setFont(QFont("Noto Sans Myanmar", 40))
        self.viewer = StrokeViewer()
        self.load_users()

        # Controls
        self.copies_combo = QComboBox()
        for i in range(1, MAX_NUM_COPIES + 1):
            self.copies_combo.addItem(str(i), i)
        self.copies_combo.setCurrentIndex(DEFAULT_NUM_COPIES - 1)

        self.check_btn = QPushButton()
        self.check_btn.setToolTip(
            "Check if each symbol has expected copies for the selected user."
        )
        self.check_btn.clicked.connect(self.run_file_check)
        self.copies_combo.currentIndexChanged.connect(self._update_check_btn_text)
        self._update_check_btn_text()

        rename_btn = QPushButton("Rename file")
        rename_btn.setToolTip(
            "Rename the selected file for the selected user."
        )
        rename_btn.clicked.connect(self.rename_selected_file)

        delete_btn = QPushButton("Delete file")
        delete_btn.setToolTip("Delete the selected file for the selected user.")
        delete_btn.clicked.connect(self.delete_selected_file)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Line # (e.g. 1029)")
        self.search_btn = QPushButton("Search")
        self.search_btn.setToolTip(
            "Jump to lowest copy for this symbol line (e.g. 1029-1 or 1029-4)."
        )
        self.search_btn.clicked.connect(self.search_symbol)
        self.search_input.returnPressed.connect(self.search_symbol)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip(
            "Reload user list and files from disk (new folders/files appear)."
        )
        refresh_btn.clicked.connect(self.refresh_data)

        # Signals
        self.user_list.currentItemChanged.connect(self.load_files)
        self.file_list.currentItemChanged.connect(self.display_file)

        # Layout
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Users"))
        left_layout.addWidget(self.user_list)

        copies_row = QHBoxLayout()
        copies_row.addWidget(QLabel("Copies"))
        copies_row.addWidget(self.copies_combo, 1)
        left_layout.addLayout(copies_row)
        left_layout.addWidget(self.check_btn)

        mid_layout = QVBoxLayout()
        search_row = QHBoxLayout()
        search_row.addWidget(self.search_input, 1)
        search_row.addWidget(self.search_btn)
        mid_layout.addLayout(search_row)
        mid_layout.addWidget(refresh_btn)
        mid_layout.addWidget(QLabel("Files"))
        mid_layout.addWidget(self.file_list)
        mid_layout.addWidget(rename_btn)
        mid_layout.addWidget(delete_btn)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.char_label)
        right_layout.addWidget(self.viewer)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(mid_layout, 1)
        main_layout.addLayout(right_layout, 3)

        self.setLayout(main_layout)

    # -------------------------
    # Load Users
    # -------------------------
    def load_users(self):
        if not os.path.exists(self.dataset_dir):
            return

        self.user_list.clear()

        # updated by SaiJack
        for user in sorted(os.listdir(self.dataset_dir)):
            user_path = os.path.join(self.dataset_dir, user)
            if not os.path.isdir(user_path):
                continue
            self.user_list.addItem(user)

    # -------------------------
    # Load Files per User
    # -------------------------
    def load_files(self):
        self.file_list.clear()

        item = self.user_list.currentItem()
        if not item:
            return

        user = item.text()
        user_path = os.path.join(self.dataset_dir, user)

        files = []
        for fname in os.listdir(user_path):
            if fname.endswith(".txt") and fname != "user_info.json":
                files.append(fname)

        # Sort like 1-1, 1-2, 2-1 ...
        files.sort(key=lambda x: [int(n) for n in re.findall(r'\d+', x)])

        for f in files:
            self.file_list.addItem(f)

    # -------------------------
    # Search File by Line Number
    # -------------------------
    def search_symbol(self):
        user_item = self.user_list.currentItem()
        if not user_item:
            QMessageBox.information(
                self, "Search", "Select a user in the list first."
            )
            return

        raw = self.search_input.text().strip()
        if not raw.isdigit():
            QMessageBox.warning(
                self,
                "Search",
                "Enter a line number (digits only, e.g. 1029).",
            )
            return

        prefix = raw
        user_path = os.path.join(self.dataset_dir, user_item.text())
        if not os.path.isdir(user_path):
            QMessageBox.warning(self, "Search", "User folder not found.")
            return

        matches = []
        for fname in os.listdir(user_path):
            m = re.match(r"^(\d+)-(\d+)\.txt$", fname)
            if m and m.group(1) == prefix:
                matches.append((int(m.group(2)), fname))

        if not matches:
            QMessageBox.warning(
                self,
                "Search",
                f"No stroke files for line {prefix} for this user.",
            )
            return

        matches.sort(key=lambda x: x[0])
        target = matches[0][1]

        self.load_files()
        for i in range(self.file_list.count()):
            if self.file_list.item(i).text() == target:
                self.file_list.setCurrentRow(i)
                return

        QMessageBox.warning(
            self,
            "Search",
            f"Found {target} on disk but it is not in the file list.",
        )

    # -------------------------
    # Refresh Data
    # -------------------------
    def refresh_data(self):
        user_item = self.user_list.currentItem()
        user_name = user_item.text() if user_item else None
        file_item = self.file_list.currentItem()
        file_name = file_item.text() if file_item else None

        self.load_users()

        restored = False
        if user_name:
            for i in range(self.user_list.count()):
                if self.user_list.item(i).text() == user_name:
                    self.user_list.setCurrentRow(i)
                    restored = True
                    break

        if not restored:
            self.file_list.clear()
            self.viewer.strokes = []
            self.viewer.update()
            self.char_label.setText("Character")
            return

        if file_name:
            for i in range(self.file_list.count()):
                if self.file_list.item(i).text() == file_name:
                    self.file_list.setCurrentRow(i)
                    break

    # -------------------------
    # Display Selected File
    # -------------------------
    def display_file(self):
        user_item = self.user_list.currentItem()
        file_item = self.file_list.currentItem()

        if not user_item or not file_item:
            return

        user = user_item.text()
        fname = file_item.text()

        filepath = os.path.join(self.dataset_dir, user, fname)

        # Load strokes
        self.viewer.load_file(filepath)

        # Extract index (e.g., 5-2.txt → 5)
        match = re.match(r"(\d+)-\d+\.txt", fname)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(self.text_lines):
                self.char_label.setText(f"{idx+1}: {self.text_lines[idx]}")
            else:
                self.char_label.setText("Unknown")
        else:
            self.char_label.setText("Unknown")


    # function to update check button label (sync with user selected dropdown item)
    def _update_check_btn_text(self):
        n = self.copies_combo.currentData()
        if n is None:
            n = self.copies_combo.currentIndex() + 1
        self.check_btn.setText(f"Check {n} copies")

    # -------------------------
    # Validate File Copies
    # -------------------------
    def run_file_check(self):
        user_item = self.user_list.currentItem()
        if not user_item:
            QMessageBox.information(
                self, "File check", "Select a user in the list first."
            )
            return

        n = int(self.copies_combo.currentData())
        name = user_item.text()
        user_path = os.path.join(self.dataset_dir, name)
        issues = validate_files_per_symbol(user_path, label=name, num_copies=n)

        if issues:
            dlg = QDialog(self)
            dlg.setObjectName("issue_list_dialog")
            dlg.setWindowTitle("Issues")
            lay = QVBoxLayout(dlg)
            text = QTextEdit()
            text.setReadOnly(True)
            text.setPlainText("\n".join(issues))
            lay.addWidget(text)
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dlg.accept)
            lay.addWidget(close_btn)
            dlg.exec_()
        else:
            QMessageBox.information(self, "File check", "No issues found.")

    # -------------------------
    # Rename File
    # -------------------------
    def rename_selected_file(self):
        user_item = self.user_list.currentItem()
        file_item = self.file_list.currentItem()
        if not user_item or not file_item:
            QMessageBox.information(
                self, "Rename file", "Select a user and a file to rename."
            )
            return

        old_name = file_item.text()
        if not re.match(r"^\d+-\d+\.txt$", old_name):
            QMessageBox.warning(self, "Rename file", "Cannot rename a non-stroke file.")
            return

        user = user_item.text()
        old_path = os.path.join(self.dataset_dir, user, old_name)
        if not os.path.isfile(old_path):
            QMessageBox.warning(self, "Rename file", "File not found.")
            return

        new_name = input_text_rename(
            self,
            "Rename file",
            "New filename:",
            old_name,
        )
        if new_name is None:
            return
        new_name = new_name.strip()
        if not new_name or new_name == old_name:
            return
        if not re.match(r"^\d+-\d+\.txt$", new_name):
            QMessageBox.warning(
                self,
                "Rename file",
                "Name must match symbolID-copyNo.txt (e.g. 1-3.txt).",
            )
            return

        new_path = os.path.join(self.dataset_dir, user, new_name)
        if os.path.exists(new_path):
            QMessageBox.warning(
                self, "Rename file", "A file with that name already exists."
            )
            return

        if not confirm_ok_cancel(
            self,
            "Rename file",
            f"Rename\n{user}/{old_name}\nto\n{new_name}?",
        ):
            return

        try:
            os.rename(old_path, new_path)
        except OSError as e:
            QMessageBox.warning(self, "Rename file", str(e))
            return

        self.load_files()
        for i in range(self.file_list.count()):
            if self.file_list.item(i).text() == new_name:
                self.file_list.setCurrentRow(i)
                break

    # -------------------------
    # Delete File
    # -------------------------
    def delete_selected_file(self):
        user_item = self.user_list.currentItem()
        file_item = self.file_list.currentItem()
        if not user_item or not file_item:
            QMessageBox.information(
                self, "Delete file", "Select a user and a file to delete."
            )
            return

        fname = file_item.text()
        if not re.match(r"^\d+-\d+\.txt$", fname):
            QMessageBox.warning(self, "Delete file", "Cannot delete a non-stroke file.")
            return

        user = user_item.text()
        path = os.path.join(self.dataset_dir, user, fname)
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Delete file", "File not found.")
            return

        if not confirm_ok_cancel(
            self,
            "Delete file",
            f"Permanently delete\n{user}/{fname}?",
        ):
            return

        try:
            os.remove(path)
        except OSError as e:
            QMessageBox.warning(self, "Delete file", str(e))
            return

        row = self.file_list.row(file_item)
        self.file_list.takeItem(row)
        self.viewer.strokes = []
        self.viewer.update()
        self.char_label.setText("Character")
        if self.file_list.count() > 0:
            self.file_list.setCurrentRow(min(row, self.file_list.count() - 1))


# -------------------------
# Utilities
# -------------------------
DEFAULT_NUM_COPIES = 4  # expected copies per symbol

MAX_NUM_COPIES = 9  # maximum possible number of copies per symbol

# function to validate (missing/extra) N copies of each symbol stroke file
def validate_files_per_symbol(user_dir, label=None, num_copies=DEFAULT_NUM_COPIES):
    """
    For one user folder: ensure each symbolID has copies (symbolID-1.txt to symbolID-num_copies.txt).
    Returns issue strings; empty means OK.
    """
    tag = label or user_dir
    if num_copies < 1 or num_copies > MAX_NUM_COPIES:
        return [f"{tag}: invalid num_copies (use 1–{MAX_NUM_COPIES})"]

    expected = set(range(1, num_copies + 1))

    if not os.path.isdir(user_dir):
        return [f"{tag}: not a directory"]

    files = [f for f in os.listdir(user_dir) if f.endswith(".txt")]
    groups = defaultdict(set)
    max_line = 0
    for f in files:
        try:
            prefix, num = f.replace(".txt", "").split("-")
            p_int = int(prefix)
            max_line = max(max_line, p_int)
            groups[prefix].add(int(num))
        except ValueError:
            continue

    # only validate symbol lines 1 to max_line
    # avoid reporting "missing" for the next file beyond the latest file
    if max_line == 0:
        max_line = 1

    issues = []
    for line_num in range(1, max_line + 1):
        prefix = str(line_num)
        nums = groups.get(prefix, set())
        missing = sorted(expected - nums)
        extra = sorted(nums - expected)
        if missing:
            issues.append(f"{tag}  line {prefix}: missing {missing}")
        if extra:
            issues.append(f"{tag}  line {prefix}: extra {extra}")
    return issues

# function to confirm ok/cancel
def confirm_ok_cancel(parent, title, text):
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setIcon(QMessageBox.Question)
    msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    msg.setDefaultButton(QMessageBox.Cancel)
    return msg.exec_() == QMessageBox.Ok

# function to input text for rename
def input_text_rename(parent, title, label, value):
    dlg = QDialog(parent)
    dlg.setObjectName("rename_file_dialog")
    dlg.setWindowTitle(title)
    lay = QVBoxLayout(dlg)
    lay.addWidget(QLabel(label))
    edit = QLineEdit(value)
    lay.addWidget(edit)
    buttons = QDialogButtonBox(
        QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dlg
    )
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    buttons.button(QDialogButtonBox.Cancel).setDefault(True)
    lay.addWidget(buttons)
    edit.returnPressed.connect(buttons.button(QDialogButtonBox.Ok).click)
    if dlg.exec_() != QDialog.Accepted:
        return None
    return edit.text()


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LipiTK Dataset Browser Tool"
    )
    parser.add_argument(
        "--dataset",
        default="dataset",
        help="Dataset folder (default: dataset)"
    )
    parser.add_argument(
        "--textfile",
        required=True,
        help="Text file (one character per line)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.textfile):
        print("Text file not found")
        sys.exit(1)

    with open(args.textfile, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    app = QApplication(sys.argv)
    app.setStyleSheet(
        """
        QMessageBox, QDialog#rename_file_dialog {
            min-width: 250px;
            min-height: 80px;
        }
        QDialog#issue_list_dialog {
            min-width: 500px;
            min-height: 300px;
        }
        """
    )

    window = MainWindow(args.dataset, lines)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
