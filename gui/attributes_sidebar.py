"""
Attributes sidebar for editing bounding box metadata.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QDateTimeEdit, 
                             QTextEdit, QGroupBox, QScrollArea)
from PyQt6.QtCore import Qt, QDateTime, pyqtSignal
from gui.magnifier_widget import MagnifierWidget


class AttributesSidebar(QWidget):
    """Sidebar widget for editing bounding box attributes."""
    
    attributes_changed = pyqtSignal(str, dict)  # Emits (box_id, attributes)
    
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
        title_label.setStyleSheet("QLabel { font-size: 14pt; font-weight: bold; padding: 5px; }")
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
        
    def show_attributes(self, box_id, attributes):
        """Show attributes for the selected box."""
        self.updating_ui = True
        self.current_box_id = box_id
        
        # Hide no selection label and show attributes
        self.no_selection_label.hide()
        scroll_area = self.findChild(QScrollArea)
        if scroll_area:
            scroll_area.show()
        
        # Update datetime
        datetime_str = attributes.get('date_time', '')
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
        comments = attributes.get('comments', '')
        self.comments_edit.setPlainText(comments)
        
        self.updating_ui = False
        
    def on_datetime_changed(self):
        """Handle date/time changes."""
        if not self.updating_ui and self.current_box_id:
            dt_string = self.datetime_edit.dateTime().toString(Qt.DateFormat.ISODate)
            self.emit_attributes_changed('date_time', dt_string)
            
    def on_comments_changed(self):
        """Handle comments changes."""
        if not self.updating_ui and self.current_box_id:
            comments = self.comments_edit.toPlainText()
            self.emit_attributes_changed('comments', comments)
            
    def clear_datetime(self):
        """Clear the datetime field."""
        if self.current_box_id:
            self.emit_attributes_changed('date_time', '')
            
    def emit_attributes_changed(self, key, value):
        """Emit attribute change with current box ID."""
        if self.current_box_id:
            # Get current attributes and update the specific key
            current_attrs = {}
            if hasattr(self, '_current_attributes'):
                current_attrs = self._current_attributes.copy()
            current_attrs[key] = value
            self._current_attributes = current_attrs
            self.attributes_changed.emit(self.current_box_id, current_attrs)
            
    def get_current_attributes(self):
        """Get the current attributes from the UI."""
        if not self.current_box_id:
            return {}
            
        attributes = {}
        
        # Get datetime
        dt_string = self.datetime_edit.dateTime().toString(Qt.DateFormat.ISODate)
        if dt_string:
            attributes['date_time'] = dt_string
            
        # Get comments
        comments = self.comments_edit.toPlainText().strip()
        if comments:
            attributes['comments'] = comments
            
        return attributes