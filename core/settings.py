import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class AppSettings:
    """Application settings."""

    gemini_api_key: str = ""
    detection_strategy: str = ""
    auto_refine_detection: bool = False
    refinement_strategy: str = ""
    refine_default_tolerance: float = 0.05
    refine_current_tolerance: float = 0.05

    # Validation settings
    warn_date_inconsistent: bool = True
    warn_non_rectangular: bool = True
    warn_nonstandard_aspect: bool = True
    standard_aspect_ratios: list[float] = field(
        default_factory=lambda: [4 / 6, 6 / 4, 5 / 7, 7 / 5]
    )
    aspect_ratio_tolerance: float = 0.02

    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to prevent setting non-existent attributes."""
        if not hasattr(self, name) and name not in self.__dataclass_fields__:
            raise AttributeError(f"Setting '{name}' does not exist")
        super().__setattr__(name, value)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppSettings":
        """Create AppSettings from dictionary, handling missing or invalid keys."""
        return cls(
            gemini_api_key=data.get("gemini_api_key", ""),
            detection_strategy=data.get("detection_strategy", ""),
            auto_refine_detection=bool(data.get("auto_refine_detection", False)),
            refinement_strategy=data.get("refinement_strategy", ""),
            refine_default_tolerance=float(data.get("refine_default_tolerance", 0.05)),
            refine_current_tolerance=float(data.get("refine_current_tolerance", 0.05)),
            warn_date_inconsistent=bool(data.get("warn_date_inconsistent", True)),
            warn_non_rectangular=bool(data.get("warn_non_rectangular", True)),
            warn_nonstandard_aspect=bool(data.get("warn_nonstandard_aspect", True)),
            standard_aspect_ratios=data.get(
                "standard_aspect_ratios", [4 / 6, 6 / 4, 5 / 7, 7 / 5]
            ),
            aspect_ratio_tolerance=float(data.get("aspect_ratio_tolerance", 0.02)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert AppSettings to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def load_from_file(cls, file_path: Optional[str] = None) -> "AppSettings":
        """Load settings from JSON file."""
        if file_path is None:
            file_path = os.path.expanduser("~/.photo_extractor_settings.json")

        if os.path.exists(file_path):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                return cls.from_dict(data)
            except (OSError, json.JSONDecodeError):
                return cls()
        return cls()

    def save_to_file(self, file_path: Optional[str] = None) -> None:
        """Save settings to JSON file."""
        if file_path is None:
            file_path = os.path.expanduser("~/.photo_extractor_settings.json")

        try:
            with open(file_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        except OSError:
            print(f"Warning: Could not save settings to {file_path}")


# Global settings instance - loaded once on import
app_settings = AppSettings.load_from_file()
