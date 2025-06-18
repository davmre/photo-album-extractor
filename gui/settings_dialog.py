"""
Settings dialog and configuration management.
"""

import json
import os

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)


class Settings:
    """Handles application settings storage and retrieval."""

    def __init__(self):
        self.settings_file = os.path.expanduser("~/.photo_extractor_settings.json")
        self.data = self.load_settings()

    def load_settings(self):
        """Load settings from JSON file."""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def save_settings(self):
        """Save settings to JSON file."""
        try:
            with open(self.settings_file, "w") as f:
                json.dump(self.data, f, indent=2)
        except OSError:
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

    def __init__(
        self, settings, detection_strategies, refinement_strategy_names, parent=None
    ):
        super().__init__(parent)
        self.settings = settings
        self.detection_strategies = detection_strategies
        self.refinement_strategy_names = refinement_strategy_names
        self.init_ui()

    def init_ui(self):
        """Initialize the settings dialog UI."""
        self.setWindowTitle("Settings")
        self.setFixedSize(450, 300)

        layout = QVBoxLayout(self)

        # Create form layout
        form_layout = QFormLayout()

        # Gemini API Key field
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("Enter your Gemini API key")
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        current_key = self.settings.get("gemini_api_key", "")
        self.api_key_edit.setText(current_key)

        form_layout.addRow("Gemini API Key:", self.api_key_edit)

        # Detection Strategy selector
        self.detection_strategy_combo = QComboBox()
        current_strategy = self.settings.get("detection_strategy", "")
        current_index = 0
        for i, strategy in enumerate(self.detection_strategies):
            self.detection_strategy_combo.addItem(strategy.name, strategy)
            if strategy.name == current_strategy:
                current_index = i
        self.detection_strategy_combo.setCurrentIndex(current_index)

        form_layout.addRow("Detection Strategy:", self.detection_strategy_combo)

        # Auto-refine checkbox
        self.auto_refine_checkbox = QCheckBox(
            "Automatically refine detected bounding boxes"
        )
        auto_refine = self.settings.get("auto_refine_detection", False)
        self.auto_refine_checkbox.setChecked(auto_refine)

        form_layout.addRow("", self.auto_refine_checkbox)

        # Refinement Strategy combo
        self.refinement_strategy_combo = QComboBox()
        current_strategy = self.settings.get("refinement_strategy", "")
        current_index = 0
        for i, strategy in enumerate(self.refinement_strategy_names):
            self.refinement_strategy_combo.addItem(strategy, strategy)
            if strategy == current_strategy:
                current_index = i
        self.refinement_strategy_combo.setCurrentIndex(current_index)

        form_layout.addRow("Refinement Strategy:", self.refinement_strategy_combo)

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
        if api_key and not api_key.startswith("AIza"):
            QMessageBox.warning(
                self,
                "Invalid API Key",
                "Gemini API keys typically start with 'AIza'. "
                "Please check your API key.",
            )
            return

        # Save all settings
        self.settings.set("gemini_api_key", api_key)

        # Save detection strategy
        selected_strategy = self.detection_strategy_combo.currentData()
        if selected_strategy:
            self.settings.set("detection_strategy", selected_strategy.name)

        # Save auto-refine setting
        self.settings.set(
            "auto_refine_detection", self.auto_refine_checkbox.isChecked()
        )

        # Save refinement strategy
        selected_strategy = self.refinement_strategy_combo.currentData()
        if selected_strategy:
            self.settings.set("refinement_strategy", selected_strategy)

        super().accept()
