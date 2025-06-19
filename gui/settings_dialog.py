"""
Settings dialog and configuration management.
"""

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

from core.settings import AppSettings
from image_processing.detection_strategies import DETECTION_STRATEGIES
from image_processing.refine_bounds import REFINEMENT_STRATEGIES


class SettingsDialog(QDialog):
    """Settings dialog for configuring application preferences."""

    def __init__(
        self,
        settings: AppSettings,
        parent=None,
    ):
        super().__init__(parent)
        self.settings = settings
        self.detection_strategies = DETECTION_STRATEGIES
        self.refinement_strategy_names = REFINEMENT_STRATEGIES.keys()
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
        self.api_key_edit.setText(self.settings.gemini_api_key)

        form_layout.addRow("Gemini API Key:", self.api_key_edit)

        # Detection Strategy selector
        self.detection_strategy_combo = QComboBox()
        current_index = 0
        for i, strategy in enumerate(self.detection_strategies):
            self.detection_strategy_combo.addItem(strategy.name, strategy)
            if strategy.name == self.settings.detection_strategy:
                current_index = i
        self.detection_strategy_combo.setCurrentIndex(current_index)

        form_layout.addRow("Detection Strategy:", self.detection_strategy_combo)

        # Auto-refine checkbox
        self.auto_refine_checkbox = QCheckBox(
            "Automatically refine detected bounding boxes"
        )
        self.auto_refine_checkbox.setChecked(self.settings.auto_refine_detection)

        form_layout.addRow("", self.auto_refine_checkbox)

        # Refinement Strategy combo
        self.refinement_strategy_combo = QComboBox()
        current_index = 0
        for i, strategy in enumerate(self.refinement_strategy_names):
            self.refinement_strategy_combo.addItem(strategy, strategy)
            if strategy == self.settings.refinement_strategy:
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
        self.settings.gemini_api_key = api_key

        # Save detection strategy
        selected_strategy = self.detection_strategy_combo.currentData()
        if selected_strategy:
            self.settings.detection_strategy = selected_strategy.name

        # Save auto-refine setting
        self.settings.auto_refine_detection = self.auto_refine_checkbox.isChecked()

        # Save refinement strategy
        selected_strategy = self.refinement_strategy_combo.currentData()
        if selected_strategy:
            self.settings.refinement_strategy = selected_strategy

        # Save to file
        self.settings.save_to_file()

        super().accept()
