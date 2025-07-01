"""
Settings dialog and configuration management.
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

from core.detection_strategies import DETECTION_STRATEGIES
from core.refinement_strategies import REFINEMENT_STRATEGIES
from core.settings import app_settings


class SettingsDialog(QDialog):
    """Settings dialog for configuring application preferences."""

    validation_settings_changed = pyqtSignal()

    def __init__(
        self,
        parent=None,
    ):
        super().__init__(parent)
        self.detection_strategies = DETECTION_STRATEGIES
        self.refinement_strategies = REFINEMENT_STRATEGIES
        self.init_ui()

    def init_ui(self):
        """Initialize the settings dialog UI."""
        self.setWindowTitle("Settings")
        self.setFixedSize(500, 450)

        layout = QVBoxLayout(self)

        # Create form layout
        form_layout = QFormLayout()

        # Gemini API Key field
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("Enter your Gemini API key")
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setText(app_settings.gemini_api_key)

        form_layout.addRow("Gemini API Key:", self.api_key_edit)

        # Detection Strategy selector
        self.detection_strategy_combo = QComboBox()
        current_index = 0
        for i, strategy in enumerate(self.detection_strategies.values()):
            self.detection_strategy_combo.addItem(strategy.name, strategy)
            if strategy.name == app_settings.detection_strategy:
                current_index = i
        self.detection_strategy_combo.setCurrentIndex(current_index)

        form_layout.addRow("Detection Strategy:", self.detection_strategy_combo)

        # Auto-refine checkbox
        self.auto_refine_checkbox = QCheckBox(
            "Automatically refine detected bounding boxes"
        )
        self.auto_refine_checkbox.setChecked(app_settings.auto_refine_detection)

        form_layout.addRow("", self.auto_refine_checkbox)

        # Refinement Strategy combo
        self.refinement_strategy_combo = QComboBox()
        current_index = 0
        for i, strategy in enumerate(self.refinement_strategies.values()):
            self.refinement_strategy_combo.addItem(strategy.name, strategy)
            if strategy.name == app_settings.refinement_strategy:
                current_index = i
        self.refinement_strategy_combo.setCurrentIndex(current_index)

        form_layout.addRow("Refinement Strategy:", self.refinement_strategy_combo)

        layout.addLayout(form_layout)

        # Validation Settings Group
        validation_group = QGroupBox("Validation Warnings")
        validation_layout = QFormLayout(validation_group)

        # Date inconsistency warning checkbox
        self.warn_date_inconsistent_checkbox = QCheckBox(
            "Warn about inconsistent dates"
        )
        self.warn_date_inconsistent_checkbox.setChecked(
            app_settings.warn_date_inconsistent
        )
        validation_layout.addRow("", self.warn_date_inconsistent_checkbox)

        # Non-rectangular warning checkbox
        self.warn_non_rectangular_checkbox = QCheckBox(
            "Warn about non-rectangular bounding boxes"
        )
        self.warn_non_rectangular_checkbox.setChecked(app_settings.warn_non_rectangular)
        validation_layout.addRow("", self.warn_non_rectangular_checkbox)

        # Nonstandard aspect ratio warning checkbox
        self.warn_nonstandard_aspect_checkbox = QCheckBox(
            "Warn about nonstandard aspect ratios"
        )
        self.warn_nonstandard_aspect_checkbox.setChecked(
            app_settings.warn_nonstandard_aspect
        )
        validation_layout.addRow("", self.warn_nonstandard_aspect_checkbox)

        # Aspect ratio tolerance
        self.aspect_ratio_tolerance_spin = QDoubleSpinBox()
        self.aspect_ratio_tolerance_spin.setRange(0.001, 0.5)
        self.aspect_ratio_tolerance_spin.setDecimals(3)
        self.aspect_ratio_tolerance_spin.setSingleStep(0.01)
        self.aspect_ratio_tolerance_spin.setValue(app_settings.aspect_ratio_tolerance)
        validation_layout.addRow(
            "Aspect Ratio Tolerance:", self.aspect_ratio_tolerance_spin
        )

        # Standard aspect ratios (as formatted text for now - could be more sophisticated)
        self.aspect_ratios_edit = QLineEdit()
        aspect_ratios_text = ", ".join(
            f"{ratio:.3f}" for ratio in app_settings.standard_aspect_ratios
        )
        self.aspect_ratios_edit.setText(aspect_ratios_text)
        self.aspect_ratios_edit.setPlaceholderText("e.g., 0.667, 1.500, 0.714, 1.400")
        validation_layout.addRow("Standard Aspect Ratios:", self.aspect_ratios_edit)

        layout.addWidget(validation_group)

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

        # Capture original validation settings before changes
        original_validation = {
            "warn_date_inconsistent": app_settings.warn_date_inconsistent,
            "warn_non_rectangular": app_settings.warn_non_rectangular,
            "warn_nonstandard_aspect": app_settings.warn_nonstandard_aspect,
            "aspect_ratio_tolerance": app_settings.aspect_ratio_tolerance,
            "standard_aspect_ratios": app_settings.standard_aspect_ratios.copy(),
        }

        # Save all settings
        app_settings.gemini_api_key = api_key

        # Save detection strategy
        selected_strategy = self.detection_strategy_combo.currentData()
        if selected_strategy:
            app_settings.detection_strategy = selected_strategy.name

        # Save auto-refine setting
        app_settings.auto_refine_detection = self.auto_refine_checkbox.isChecked()

        # Save refinement strategy
        selected_strategy = self.refinement_strategy_combo.currentData()
        if selected_strategy:
            app_settings.refinement_strategy = selected_strategy.name

        # Save validation settings
        app_settings.warn_date_inconsistent = (
            self.warn_date_inconsistent_checkbox.isChecked()
        )
        app_settings.warn_non_rectangular = (
            self.warn_non_rectangular_checkbox.isChecked()
        )
        app_settings.warn_nonstandard_aspect = (
            self.warn_nonstandard_aspect_checkbox.isChecked()
        )
        app_settings.aspect_ratio_tolerance = self.aspect_ratio_tolerance_spin.value()

        # Parse and save standard aspect ratios
        try:
            aspect_ratios_text = self.aspect_ratios_edit.text().strip()
            if aspect_ratios_text:
                aspect_ratios = [
                    float(x.strip()) for x in aspect_ratios_text.split(",")
                ]
                app_settings.standard_aspect_ratios = aspect_ratios
            else:
                # Empty field - use defaults
                app_settings.standard_aspect_ratios = [4 / 6, 6 / 4, 5 / 7, 7 / 5]
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid Aspect Ratios",
                "Please enter valid aspect ratios separated by commas (e.g., 0.667, 1.500)",
            )
            return

        # Save to file
        app_settings.save_to_file()

        # Check if validation settings actually changed
        validation_changed = (
            original_validation["warn_date_inconsistent"]
            != app_settings.warn_date_inconsistent
            or original_validation["warn_non_rectangular"]
            != app_settings.warn_non_rectangular
            or original_validation["warn_nonstandard_aspect"]
            != app_settings.warn_nonstandard_aspect
            or original_validation["aspect_ratio_tolerance"]
            != app_settings.aspect_ratio_tolerance
            or original_validation["standard_aspect_ratios"]
            != app_settings.standard_aspect_ratios
        )

        if validation_changed:
            self.validation_settings_changed.emit()

        super().accept()
