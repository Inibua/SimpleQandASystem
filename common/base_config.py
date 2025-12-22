# ----------------------------------------------------------------------
# Helper / core implementation
# ----------------------------------------------------------------------
import json
from pathlib import Path
from typing import Mapping, Dict, Any, MutableMapping
from paths import ROOT_DIR


class ConfigError(RuntimeError):
    """Raised when a JSON config file cannot be parsed or is invalid."""
    pass


class BaseConfig:
    """
        Base class for all concrete config objects.

        Sub‑classes **must** provide two class attributes:

        * ``_FILENAME`` – the JSON file name (relative to the ``configs/`` folder)
        * ``_REQUIRED`` – a mapping ``{key_name: default_value}`` where
          ``default_value`` may be ``None`` if the key has no sensible default.
          The presence of the key is mandatory; if a default is supplied it is
          used when the key is missing, otherwise a ``ConfigError`` is raised.
        """

    # folder that holds all config *.json files – change if you relocate them
    _CONFIG_ROOT = Path(ROOT_DIR, "configs")

    # ------------------------------------------------------------------
    # To be overridden by subclasses
    # ------------------------------------------------------------------
    _FILENAME: str = ""  # e.g. "indexer.json"
    _REQUIRED: Mapping[str, Any] = {}  # e.g. {"data_directory": None}

    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """
        Load the JSON file, validate required keys, and expose everything as
        attributes on ``self``.
        """
        json_path = self._config_path()
        raw_cfg = self._load_json(json_path)
        cfg = self._apply_defaults_and_validate(raw_cfg)

        # expose every key as an attribute (including extra keys)
        for k, v in cfg.items():
            setattr(self, k, v)

        # keep a reference to the whole dict – handy for debugging / logging
        self._raw: Dict[str, Any] = cfg

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _config_path(self) -> Path:
        """Return the absolute path to the JSON file for this config."""
        if not self._FILENAME:
            raise ConfigError(
                f"{self.__class__.__name__} does not define a _FILENAME."
            )
        path = self._CONFIG_ROOT / self._FILENAME
        if not path.is_file():
            raise ConfigError(f"Config file not found: {path}")
        return path

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        """Read a JSON file and return a dict; raise ConfigError on failure."""
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            raise ConfigError(f"Unable to read config {path}: {exc}") from exc
        return data

    def _apply_defaults_and_validate(self, raw: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Merge ``raw`` with defaults, ensure required keys exist,
        and return a plain ``dict``.
        """
        cfg: MutableMapping[str, Any] = dict(raw)  # shallow copy

        missing: list[str] = []
        for key, default in self._REQUIRED.items():
            if key not in cfg:
                if default is not None:
                    cfg[key] = default
                else:
                    missing.append(key)

        if missing:
            raise ConfigError(
                f"The following required keys are missing in {self._FILENAME}: "
                f"{', '.join(missing)}"
            )

        # No strict prohibition of extra keys – they stay in the dict.
        # If you ever want to forbid them, uncomment the next lines:
        #
        # extra = set(cfg) - set(self._REQUIRED)
        # if extra:
        #     raise ConfigError(
        #         f"Unexpected keys in {self._FILENAME}: {', '.join(extra)}"
        #     )
        return dict(cfg)  # return a plain dict

    # ------------------------------------------------------------------
    # Public helpers (optional but convenient)
    # ------------------------------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        """Return the whole configuration as a plain dict."""
        return dict(self._raw)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        keys = ", ".join(sorted(self._raw.keys()))
        return f"<{cls} keys=[{keys}]>"