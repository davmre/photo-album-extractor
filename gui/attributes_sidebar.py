"""
Attributes sidebar for editing bounding box metadata.
"""

from PyQt6.QtCore import QDateTime, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDateTimeEdit,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.photo_types import BoundingBoxData, PhotoAttributes
from gui.magnifier_widget import MagnifierWidget


class AttributesSidebar(QWidget):
    """Sidebar widget for editing bounding box attributes."""

    attributes_changed = pyqtSignal(str, PhotoAttributes)  # Emits (box_id, attributes)
    coordinates_changed = pyqtSignal(str, list)  # Emits (box_id, coordinates)

    def __init__(self):
        super().__init__()
        self.current_box_id = None
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
        datetime_group = QGroupBox("Date & Time")
        datetime_layout = QVBoxLayout(datetime_group)

        self.datetime_edit = QDateTimeEdit()
        self.datetime_edit.setCalendarPopup(True)
        self.datetime_edit.setDateTime(QDateTime.currentDateTime())
        self.datetime_edit.setDisplayFormat("M/d/yyyy h:mm")
        self.datetime_edit.dateTimeChanged.connect(self.on_datetime_changed)
        datetime_layout.addWidget(self.datetime_edit)

        # Clear date button
        self.clear_date_label = QLabel("<a href='#'>Clear date</a>")
        self.clear_date_label.setOpenExternalLinks(False)
        self.clear_date_label.linkActivated.connect(self.clear_datetime)
        self.clear_date_label.setStyleSheet("QLabel { color: #0066cc; }")
        datetime_layout.addWidget(self.clear_date_label)

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

        # Coordinates group
        coordinates_group = QGroupBox("Corner Coordinates")
        coordinates_layout = QVBoxLayout(coordinates_group)

        # Create coordinate input fields for each corner
        self.coordinate_spinboxes = []
        corner_labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]

        for i, label in enumerate(corner_labels):
            corner_layout = QVBoxLayout()
            corner_label = QLabel(f"{label}:")
            corner_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
            corner_layout.addWidget(corner_label)

            # X coordinate
            x_layout = QHBoxLayout()
            x_layout.addWidget(QLabel("X:"))
            x_spinbox = QSpinBox()
            x_spinbox.setRange(-9999, 9999)
            x_spinbox.valueChanged.connect(
                lambda value, corner=i, coord="x": self.on_coordinate_changed(
                    corner, coord, value
                )
            )
            x_layout.addWidget(x_spinbox)
            corner_layout.addLayout(x_layout)

            # Y coordinate
            y_layout = QHBoxLayout()
            y_layout.addWidget(QLabel("Y:"))
            y_spinbox = QSpinBox()
            y_spinbox.setRange(-9999, 9999)
            y_spinbox.valueChanged.connect(
                lambda value, corner=i, coord="y": self.on_coordinate_changed(
                    corner, coord, value
                )
            )
            y_layout.addWidget(y_spinbox)
            corner_layout.addLayout(y_layout)

            self.coordinate_spinboxes.append((x_spinbox, y_spinbox))
            coordinates_layout.addLayout(corner_layout)

        content_layout.addWidget(coordinates_group)

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
        self.current_box_id = None
        scroll_area = self.findChild(QScrollArea)
        if scroll_area:
            scroll_area.hide()
        self.no_selection_label.show()

    def show_box_data(self, box_data: BoundingBoxData):
        """Show attributes for the selected box."""
        self.updating_ui = True
        self.current_box_id = box_data.box_id
        self._current_attributes = box_data.attributes

        # Hide no selection label and show attributes
        self.no_selection_label.hide()
        scroll_area = self.findChild(QScrollArea)
        if scroll_area:
            scroll_area.show()

        # Update datetime
        datetime_str = box_data.attributes.date_time
        if datetime_str:
            try:
                dt = QDateTime.fromString(datetime_str, Qt.DateFormat.ISODate)
                if dt.isValid():
                    self.datetime_edit.setDateTime(dt)
                else:
                    self.datetime_edit.setDateTime(QDateTime.currentDateTime())
            except:
                self.datetime_edit.setDateTime(QDateTime.currentDateTime())
        else:
            self.datetime_edit.setDateTime(QDateTime.currentDateTime())

        # Update comments
        current_comments = self.comments_edit.toPlainText()
        new_comments = box_data.attributes.comments
        if current_comments != new_comments:
            self.comments_edit.setPlainText(box_data.attributes.comments)

        try:
            for i, (x_spinbox, y_spinbox) in enumerate(self.coordinate_spinboxes):
                if i < len(box_data.corners):
                    x_spinbox.setValue(box_data.corners[i][0])
                    y_spinbox.setValue(box_data.corners[i][1])
        finally:
            self.updating_ui = False

    def on_datetime_changed(self):
        """Handle date/time changes."""
        if not self.updating_ui and self.current_box_id:
            dt_string = self.datetime_edit.dateTime().toString(Qt.DateFormat.ISODate)
            self.emit_attributes_changed("date_time", dt_string)

    def on_comments_changed(self):
        """Handle comments changes."""
        if not self.updating_ui and self.current_box_id:
            comments = self.comments_edit.toPlainText()
            self.emit_attributes_changed("comments", comments)

    def clear_datetime(self):
        """Clear the datetime field."""
        if self.current_box_id:
            self.emit_attributes_changed("date_time", "")

    def emit_attributes_changed(self, key: str, value: str):
        """Emit attribute change with current box ID."""
        if self.current_box_id:
            # Update current attributes
            if hasattr(self, "_current_attributes"):
                attrs = self._current_attributes
            else:
                attrs = PhotoAttributes()

            # Set the specific attribute
            setattr(attrs, key, value)
            self._current_attributes = attrs
            self.attributes_changed.emit(self.current_box_id, attrs)

    def get_current_attributes(self) -> PhotoAttributes:
        """Get the current attributes from the UI."""
        if not self.current_box_id:
            return PhotoAttributes()

        # Get datetime
        dt_string = self.datetime_edit.dateTime().toString(Qt.DateFormat.ISODate)

        # Get comments
        comments = self.comments_edit.toPlainText().strip()

        return PhotoAttributes(date_time=dt_string, comments=comments)

    def on_coordinate_changed(self, corner_index, coord_type, value):
        """Handle coordinate changes."""
        if not self.updating_ui and self.current_box_id:
            # Get current coordinates from all spinboxes
            coordinates = []
            for x_spinbox, y_spinbox in self.coordinate_spinboxes:
                x_val = x_spinbox.value()
                y_val = y_spinbox.value()
                coordinates.append([float(x_val), float(y_val)])

            # Focus magnifier on the corner being edited
            if corner_index < len(coordinates):
                corner_pos = coordinates[corner_index]
                self.magnifier.focus_on_corner(corner_pos)

            # Emit coordinate change signal
            self.coordinates_changed.emit(self.current_box_id, coordinates)
