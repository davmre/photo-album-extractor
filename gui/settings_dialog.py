"""
Settings dialog and configuration management.
"""

import os
import json
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLineEdit, 
                             QDialogButtonBox, QMessageBox)


class Settings:
    """Handles application settings storage and retrieval."""
    
    def __init__(self):
        self.settings_file = os.path.expanduser("~/.photo_extractor_settings.json")
        self.data = self.load_settings()
        
    def load_settings(self):
        """Load settings from JSON file."""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
        
    def save_settings(self):
        """Save settings to JSON file."""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except IOError:
            print(f"Warning: Could not save settings to {self.settings_file}")
            
    def get(self, key, default=None):
        """Get a setting value."""
        return self.data.get(key, default)
        
    def set(self, key, value):
        """Set a setting value and save."""
        self.data[key] = value
        self.save_settings()


class SettingsDialog(QDialog):
    """Settings dialog for configuring application preferences."""
    
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.init_ui()
        
    def init_ui(self):
        """Initialize the settings dialog UI."""
        self.setWindowTitle("Settings")
        self.setFixedSize(400, 200)
        
        layout = QVBoxLayout(self)
        
        # Create form layout
        form_layout = QFormLayout()
        
        # Gemini API Key field
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("Enter your Gemini API key")
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        current_key = self.settings.get('gemini_api_key', '')
        self.api_key_edit.setText(current_key)
        
        form_layout.addRow("Gemini API Key:", self.api_key_edit)
        
        layout.addLayout(form_layout)
        
        # Add some spacing
        layout.addStretch()
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def accept(self):
        """Save settings when OK is clicked."""
        api_key = self.api_key_edit.text().strip()
        
        # Basic validation for Gemini API key format
        if api_key and not api_key.startswith('AIza'):
            QMessageBox.warning(self, "Invalid API Key", 
                              "Gemini API keys typically start with 'AIza'. "
                              "Please check your API key.")
            return
            
        self.settings.set('gemini_api_key', api_key)
        super().accept()