"""Engine for translating parameter names across ML frameworks and dialects without hallucinations."""

from __future__ import annotations

import json
import os
from typing import Any

from ml_switcheroo_ir.validator import DEFAULT_SNAPSHOT_DIR


class ParameterTranslationEngine:
    """Translates parameter and attribute names across frameworks based on concept maps.

    Ensures zero hallucinated parameters by validating that every translated
    parameter corresponds to an empirical, grounded framework parameter role.
    """

    def __init__(
        self,
        translations: dict[str, Any] | None = None,
        concept_map_path: str | None = None,
    ) -> None:
        """Initialize translation engine with parameter translation mapping.

        Args:
            translations (Optional[dict[str, Any]]): Pre-loaded parameter translations mapping.
            concept_map_path (Optional[str]): Path to concept_map.json file.
        """
        self.translations: dict[str, Any] = translations or {}
        if not self.translations:
            self._load_default_translations(concept_map_path)

    def _load_default_translations(self, path: str | None = None) -> None:
        """Load default translations from concept_map.json in snapshot directory or package root.

        Args:
            path (Optional[str]): Optional path override.
        """
        if path is not None:
            candidates = [path]
        else:
            candidates = [
                os.path.join(DEFAULT_SNAPSHOT_DIR, "concept_map.json"),
                os.path.join(os.path.dirname(DEFAULT_SNAPSHOT_DIR), "concept_map.json"),
            ]
        for cand in candidates:
            if cand and os.path.isfile(cand):
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "_parameter_translations" in data and isinstance(
                        data["_parameter_translations"], dict
                    ):
                        self.translations.update(data["_parameter_translations"])
                        break
                except (json.JSONDecodeError, OSError):
                    pass

    def translate_parameter(
        self,
        operation: str,
        param_name: str,
        source_framework: str,
        target_framework: str,
    ) -> str:
        """Translate a single parameter name from source framework to target framework.

        Args:
            operation (str): Canonical operation name (e.g. 'matmul', 'reduction', 'convolution', 'normalization').
            param_name (str): Parameter name in source framework.
            source_framework (str): Source framework identifier (e.g. 'torch', 'jax', 'tf', 'stablehlo', 'numpy').
            target_framework (str): Target framework identifier.

        Returns:
            str: Translated parameter name in target framework.

        Raises:
            KeyError: If operation or parameter role is not found in empirical translations.
            ValueError: If source or target framework has no mapping for the parameter role.
        """
        if operation not in self.translations:
            raise KeyError(
                f"No parameter translations defined for operation '{operation}'."
            )

        roles = self.translations[operation].get("roles", {})
        for role_name, fw_mappings in roles.items():
            src_params = fw_mappings.get(source_framework, [])
            if param_name in src_params:
                target_params = fw_mappings.get(target_framework, [])
                if not target_params:
                    raise ValueError(
                        f"Target framework '{target_framework}' does not support role '{role_name}' for operation '{operation}'."
                    )
                return str(target_params[0])

        raise KeyError(
            f"Parameter '{param_name}' is not recognized as a valid '{source_framework}' parameter for '{operation}'."
        )

    def translate_attributes(
        self,
        operation: str,
        attributes: dict[str, Any],
        source_framework: str,
        target_framework: str,
        allow_passthrough: bool = False,
    ) -> dict[str, Any]:
        """Translate all attributes/keyword arguments from source framework to target framework.

        Args:
            operation (str): Canonical operation name.
            attributes (dict[str, Any]): Dictionary of parameter names and values.
            source_framework (str): Source framework identifier.
            target_framework (str): Target framework identifier.
            allow_passthrough (bool): If True, parameters without translation are kept; if False,
                unrecognized parameters raise ValueError (zero-hallucination policy).

        Returns:
            dict[str, Any]: Translated attributes dictionary.

        Raises:
            ValueError: If unknown parameter is encountered and allow_passthrough is False.
        """
        result: dict[str, Any] = {}
        for k, v in attributes.items():
            try:
                translated_key = self.translate_parameter(
                    operation=operation,
                    param_name=k,
                    source_framework=source_framework,
                    target_framework=target_framework,
                )
                result[translated_key] = v
            except (KeyError, ValueError) as err:
                if allow_passthrough:
                    result[k] = v
                else:
                    raise ValueError(
                        f"Ungrounded parameter '{k}' during translation from '{source_framework}' to '{target_framework}': {err}"
                    ) from err
        return result
