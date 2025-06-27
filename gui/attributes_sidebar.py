"""
Attributes sidebar for editing bounding box metadata.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.bounding_box_data import BoundingBoxData, PhotoAttributes
from core.validation_utils import Severity, validate_bounding_box
from gui.magnifier_widget import MagnifierWidget


class SelectAllLineEdit(QLineEdit):
    """QLineEdit that automatically selects all text when it gains focus."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._should_select_all_on_mouse_press = False

    def focusInEvent(self, event):
        """Override to set up select-all behavior when widget gains focus."""
        super().focusInEvent(event)
        # Flag that the next mouse press should select all (if there's text)
        if self.text():
            self._should_select_all_on_mouse_press = True

    def mousePressEvent(self, event):
        """Override to implement select-all-on-first-click behavior."""
        if self._should_select_all_on_mouse_press and self.text():
            # Select all text instead of positioning cursor at click location
            self.selectAll()
            self._should_select_all_on_mouse_press = False
        else:
            # Normal mouse press behavior (position cursor at click)
            super().mousePressEvent(event)

    def keyPressEvent(self, event):
        """Override to cancel select-all behavior when user types."""
        # Any key press means user is actively editing, so disable select-all
        self._should_select_all_on_mouse_press = False
        super().keyPressEvent(event)


class AttributesSidebar(QWidget):
    """Sidebar widget for editing bounding box attributes."""

    attributes_changed = pyqtSignal(
        str, str, PhotoAttributes
    )  # Emits (box_id, key_changed, attributes)

    def __init__(self):
        super().__init__()
        self.current_box = None
        self.updating_ui = False  # Flag to prevent recursion

        # Set up the widget
        self.setMaximumWidth(300)
        self.setMinimumWidth(250)

        # Create layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Title
        title_label = QLabel("Photo Attributes")
        title_label.setStyleSheet(
            "QLabel { font-size: 14pt; font-weight: bold; padding: 5px; }"
        )
        main_layout.addWidget(title_label)

        # Create scroll area for attributes
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Create content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(5, 5, 5, 5)
        content_layout.setSpacing(10)

        # Date/Time group
        datetime_group = QGroupBox("Date")
        datetime_layout = QVBoxLayout(datetime_group)

        hint_layout = QHBoxLayout()
        self.hint_label = QLabel("Hint:")
        hint_layout.addWidget(self.hint_label)

        self.date_hint_edit = SelectAllLineEdit()
        self.date_hint_edit.setPlaceholderText("e.g., May 2021, 1999-08-14, 2023")
        self.date_hint_edit.editingFinished.connect(self.on_datetime_changed)
        hint_layout.addWidget(self.date_hint_edit)

        datetime_layout.addLayout(hint_layout)

        exif_layout = QHBoxLayout()
        self.inferred_exif_label = QLabel("Inferred:")
        exif_layout.addWidget(self.inferred_exif_label)

        self.inferred_exif = QLabel()
        exif_layout.addWidget(self.inferred_exif)

        datetime_layout.addLayout(exif_layout)

        content_layout.addWidget(datetime_group)

        # Comments group
        comments_group = QGroupBox("Comments")
        comments_layout = QVBoxLayout(comments_group)

        self.comments_edit = QTextEdit()
        self.comments_edit.setMaximumHeight(150)
        self.comments_edit.setPlaceholderText("Add comments about this photo...")
        self.comments_edit.textChanged.connect(self.on_comments_changed)
        comments_layout.addWidget(self.comments_edit)

        content_layout.addWidget(comments_group)

        # Validation group
        self.validation_group = QGroupBox("Validation")
        validation_layout = QVBoxLayout(self.validation_group)

        self.validation_label = QLabel("No issues")
        self.validation_label.setWordWrap(True)
        self.validation_label.setStyleSheet("color: #666; padding: 5px;")
        validation_layout.addWidget(self.validation_label)

        content_layout.addWidget(self.validation_group)

        # Add stretch to push content to top
        content_layout.addStretch()

        # Set content widget to scroll area
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)

        # No selection state
        self.no_selection_label = QLabel("Select a bounding box to edit its attributes")
        self.no_selection_label.setStyleSheet("QLabel { color: #666; padding: 20px; }")
        self.no_selection_label.setWordWrap(True)
        self.no_selection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.no_selection_label)

        # Add magnifier widget at the bottom
        self.magnifier = MagnifierWidget(zoom_factor=6, size=200)
        main_layout.addWidget(self.magnifier)

        # Initially show no selection state
        self.show_no_selection()

    def show_no_selection(self):
        """Show the no selection state."""
        self.current_box = None
        scroll_area = self.findChild(QScrollArea)
        if scroll_area:
            scroll_area.hide()
        self.no_selection_label.show()

    def set_box_data(self, box_data: BoundingBoxData):
        """Show attributes for the selected box."""
        self.updating_ui = True
        self.current_box = box_data
        self._current_attributes = box_data.attributes

        # Hide no selection label and show attributes
        self.no_selection_label.hide()
        scroll_area = self.findChild(QScrollArea)
        if scroll_area:
            scroll_area.show()

        # Update datetime
        datetime_str = box_data.attributes.date_hint
        self.date_hint_edit.setText(datetime_str)
        self.inferred_exif.setText(box_data.attributes.exif_date)

        # Update comments
        current_comments = self.comments_edit.toPlainText()
        new_comments = box_data.attributes.comments
        if current_comments != new_comments:
            self.comments_edit.setPlainText(box_data.attributes.comments)

        self.updating_ui = False

        # Update validation display
        self.update_validation_display(box_data)

    def on_datetime_changed(self):
        """Handle date/time changes."""
        if not self.updating_ui and self.current_box:
            user_input = self.date_hint_edit.text().strip()
            print("datetime changed", self._current_attributes.date_hint, user_input)
            if not user_input:
                # Empty input - clear the date
                self.emit_attributes_changed("date_hint", "")
                return

            # parsed_dt = date_utils.parse_flexible_date(user_input) or ""
            # self.inferred_exif.setText(parsed_dt)

            self.emit_attributes_changed("date_hint", user_input.strip())
            # self.emit_attributes_changed("exif_date", parsed_dt)

            # Update validation display
            current_box = self.get_current_box_data()
            if current_box:
                self.update_validation_display(current_box)

    def on_comments_changed(self):
        """Handle comments changes."""
        if not self.updating_ui and self.current_box:
            comments = self.comments_edit.toPlainText()
            self.emit_attributes_changed("comments", comments)

            # Update validation display
            current_box = self.get_current_box_data()
            if current_box:
                self.update_validation_display(current_box)

    def emit_attributes_changed(self, key: str, value: str):
        """Emit attribute change with current box ID."""
        if self.current_box:
            # Update current attributes
            if hasattr(self, "_current_attributes"):
                attrs = self._current_attributes
            else:
                attrs = PhotoAttributes()

            # Set the specific attribute
            setattr(attrs, key, value)
            self._current_attributes = attrs
            self.attributes_changed.emit(self.current_box.box_id, key, attrs)

    def get_current_attributes(self) -> PhotoAttributes:
        """Get the current attributes from the UI."""
        if not self.current_box:
            return PhotoAttributes()
        # return self._current_attributes.copy()
        return PhotoAttributes(
            date_hint=self.date_hint_edit.text().strip(),
            exif_date=self.inferred_exif.text(),
            comments=self.comments_edit.toPlainText(),
            date_inconsistent=getattr(
                self._current_attributes, "date_inconsistent", False
            ),
        )

    def get_current_box_data(self) -> BoundingBoxData | None:
        """Get current bounding box data from UI state."""
        if not self.current_box:
            return None

        return BoundingBoxData(
            corners=self.current_box.corners,
            box_id=self.current_box.box_id,
            attributes=self.get_current_attributes(),
        )

    def update_validation_display(self, box_data: BoundingBoxData):
        """Update the validation display for the current bounding box."""
        issues = validate_bounding_box(box_data)

        if not issues:
            self.validation_label.setText("No issues")
            self.validation_label.setStyleSheet("color: #666; padding: 5px;")
            return

        # Sort issues by severity (errors first)
        issues.sort(key=lambda x: x.severity == Severity.ERROR, reverse=True)

        # Build display text with icons
        display_lines = []
        for issue in issues:
            if issue.severity == Severity.ERROR:
                icon = "🚨"
            else:
                icon = "⚠️"

            display_lines.append(f"{icon} {issue.message}")

        display_text = "\n".join(display_lines)
        self.validation_label.setText(display_text)

        # Set color based on highest severity
        has_errors = any(issue.severity == Severity.ERROR for issue in issues)
        color = "#cc0000" if has_errors else "#ff8800"
        self.validation_label.setStyleSheet(f"color: {color}; padding: 5px;")
