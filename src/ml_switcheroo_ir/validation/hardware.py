"""Hardware ISA, RDNA3 VOPD pairing, NVIDIA SASS scoreboarding, and WebGPU WGSL validation."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ml_switcheroo_ir.graph import LogicalNode
from ml_switcheroo_ir.schema.ghost import (
    WGSL_COMPUTE_BUILTINS,
    WGSL_MUTATING_OPS,
)
from ml_switcheroo_ir.schema.rdna_registry import (
    RDNA_REGISTRY,
    RDNA_TO_VOPD_MAP,
    RDNA_VOPD_SLOTS,
)
from ml_switcheroo_ir.schema.sass_registry import SASS_PIPELINE_LATENCIES
from ml_switcheroo_ir.validation.core import ValidationError, ValidationLevel

__all__ = [
    "_extract_dst_vgpr",
    "_extract_sass_registers",
    "_extract_vgpr_indices",
    "validate_isa_instruction_node",
    "validate_sass_scoreboarding_sequence",
    "validate_vopd_pairing",
]


_VGPR_ATTR_PATTERN = re.compile(r"^(?:v|vgpr)(\d+)$|^(\d+)$", re.IGNORECASE)
_VGPR_PORT_PATTERN = re.compile(r"^(?:v|vgpr)(\d+)$", re.IGNORECASE)


def _extract_vgpr_indices(node: LogicalNode) -> list[int]:
    """Extract VGPR operand integer indices from a node.

    Explicit ``vgpr_operands`` and ``src_vgprs`` attributes accept integers or
    ``v<N>`` / ``vgpr<N>`` strings, while ``node.inputs`` only match explicit
    ``v<N>`` or ``vgpr<N>`` register names so numeric node IDs are not misparsed.

    Args:
        node (LogicalNode): The node to extract VGPR indices from.

    Returns:
        list[int]: List of VGPR register indices.
    """
    indices: list[int] = []
    for key in ("vgpr_operands", "src_vgprs"):
        val = node.attributes.get(key)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, int) and not isinstance(item, bool):
                    indices.append(item)
                else:
                    m = _VGPR_ATTR_PATTERN.match(str(item).strip())
                    if m:
                        indices.append(int(m.group(1) or m.group(2)))

    for inp in node.inputs:
        m = _VGPR_PORT_PATTERN.match(str(inp).strip())
        if m:
            indices.append(int(m.group(1)))

    return indices


def _extract_dst_vgpr(node: LogicalNode) -> int | None:
    """Extract destination VGPR index from a node.

    Args:
        node (LogicalNode): The node to extract destination VGPR from.

    Returns:
        int | None: The destination VGPR index, or None if unspecified.
    """
    dst = node.attributes.get("dst_vgpr")
    if dst is None:
        dst = node.attributes.get("dst")
    if isinstance(dst, int) and not isinstance(dst, bool):
        return dst
    if dst is not None:
        m = _VGPR_ATTR_PATTERN.match(str(dst).strip())
        if m:
            return int(m.group(1) or m.group(2))
    for out in node.outputs:
        m = _VGPR_PORT_PATTERN.match(str(out).strip())
        if m:
            return int(m.group(1))
    return None


def _extract_sass_registers(node: LogicalNode, role: str = "src") -> list[str]:
    """Extract SASS register operands (e.g., 'R0', 'R1') from a node.

    Args:
        node (LogicalNode): The SASS instruction node.
        role (str): Operand role to extract ('src' or 'dst'). Defaults to 'src'.

    Returns:
        list[str]: Normalized register names.
    """
    primary_key, alt_key, ports = (
        ("dst_reg", "dst", node.outputs) if role == "dst" else ("src_regs", "srcs", node.inputs)
    )
    raw = node.attributes.get(primary_key)
    if raw is None:
        raw = node.attributes.get(alt_key)

    regs: list[str] = []
    if isinstance(raw, list):
        regs.extend(str(item).upper() for item in raw)
    elif raw is not None:
        regs.append(str(raw).upper())

    for port in ports:
        regs.extend(reg.upper() for reg in re.findall(r"^R\d+$", str(port), re.IGNORECASE))

    return list(dict.fromkeys(regs))


def _extract_vopd_literals(node: LogicalNode) -> set[str]:
    """Extract 32-bit literal immediate tokens referenced by a GFX11 VOPD instruction node.

    Args:
        node (LogicalNode): VOPD Slot X or Slot Y candidate node.

    Returns:
        set[str]: Normalized literal constant strings referenced by ``node``.
    """
    literals: set[str] = set()
    for key in ("literal", "imm", "imm32"):
        val = node.attributes.get(key)
        if isinstance(val, list):
            for elem in val:
                if elem is not None and not isinstance(elem, bool):
                    literals.add(str(elem).strip().lower())
        elif val is not None and not isinstance(val, bool):
            literals.add(str(val).strip().lower())
    for inp in node.inputs:
        tok = str(inp).strip().lower()
        if tok.startswith("0x"):
            literals.add(tok)
    return literals


def _check_vopd_slot(node: LogicalNode, slot_label: str) -> list[ValidationError]:
    """Verify domain and Slot X/Y eligibility for one VOPD instruction operand.

    Args:
        node (LogicalNode): Instruction node (`opX` or `opY`).
        slot_label (str): ``'X'`` or ``'Y'``.

    Returns:
        list[ValidationError]: Domain or slot eligibility errors.
    """
    errors: list[ValidationError] = []
    param_label = f"op{slot_label}"
    if node.domain != "amd_rdna":
        errors.append(
            ValidationError(
                node_id=node.id,
                attribute="domain",
                message=f"{param_label} domain must be 'amd_rdna', got '{node.domain}'.",
                level=ValidationLevel.ERROR,
            )
        )
    canon = RDNA_TO_VOPD_MAP.get(node.op_type, node.op_type)
    allowed_slot = RDNA_VOPD_SLOTS.get(canon, "")
    if canon not in RDNA_REGISTRY or not allowed_slot:
        errors.append(
            ValidationError(
                node_id=node.id,
                attribute="vopd_opcode",
                message=f"Opcode '{node.op_type}' is not a valid RDNA3 VOPD instruction.",
                level=ValidationLevel.ERROR,
            )
        )
    elif slot_label not in allowed_slot and allowed_slot != "BOTH":
        errors.append(
            ValidationError(
                node_id=node.id,
                attribute="vopd_slot",
                message=f"Opcode '{node.op_type}' cannot be issued in VOPD Slot {slot_label}.",
                level=ValidationLevel.ERROR,
            )
        )
    return errors


def validate_vopd_pairing(
    opX: LogicalNode,
    opY: LogicalNode,
) -> list[ValidationError]:
    """Validate AMD RDNA3 / GFX11 dual-issue VOPD instruction pairing rules.

    Checks opcode pairing matrix, Slot X/Y compatibility, GFX11 destination VGPR
    parity (``vdst_x % 2 != vdst_y % 2``), single 32-bit literal bus sharing, and
    4-bank source VGPR exclusivity.

    Args:
        opX (LogicalNode): Instruction for VOPD Slot X.
        opY (LogicalNode): Instruction for VOPD Slot Y.

    Returns:
        list[ValidationError]: Detected VOPD pairing violations.
    """
    errors: list[ValidationError] = []
    errors.extend(_check_vopd_slot(opX, "X"))
    errors.extend(_check_vopd_slot(opY, "Y"))

    dst_x = _extract_dst_vgpr(opX)
    dst_y = _extract_dst_vgpr(opY)
    if dst_x is not None and dst_y is not None:
        if dst_x == dst_y:
            errors.append(
                ValidationError(
                    node_id=opY.id,
                    attribute="vopd_destination",
                    message=f"VOPD destination conflict: opX and opY both write to VGPR v{dst_x}.",
                    level=ValidationLevel.ERROR,
                )
            )
        elif (dst_x % 2) == (dst_y % 2):
            errors.append(
                ValidationError(
                    node_id=opY.id,
                    attribute="vopd_destination",
                    message=(
                        f"GFX11 VOPD destination parity conflict: v{dst_x} and v{dst_y} "
                        "share even/odd parity (requires vdst_x % 2 != vdst_y % 2)."
                    ),
                    level=ValidationLevel.ERROR,
                )
            )

    literals_union = _extract_vopd_literals(opX) | _extract_vopd_literals(opY)
    if len(literals_union) > 1:
        errors.append(
            ValidationError(
                node_id=opY.id,
                attribute="vopd_literal",
                message=(
                    f"GFX11 VOPD single-literal violation: dual-issued instructions reference "
                    f"{len(literals_union)} distinct 32-bit literals {sorted(literals_union)} (max 1)."
                ),
                level=ValidationLevel.ERROR,
            )
        )

    src_x = _extract_vgpr_indices(opX)
    src_y = _extract_vgpr_indices(opY)

    for bank in range(4):
        bank_x_regs = {r for r in src_x if r % 4 == bank}
        bank_y_regs = {r for r in src_y if r % 4 == bank}
        if bank_x_regs and bank_y_regs and (bank_x_regs != bank_y_regs or len(bank_x_regs) > 1 or len(bank_y_regs) > 1):
            errors.append(
                ValidationError(
                    node_id=opX.id,
                    attribute="vopd_bank_conflict",
                    message=(
                        f"VOPD register bank conflict on bank {bank}: "
                        f"opX reads VGPRs {sorted(bank_x_regs)} and opY reads VGPRs {sorted(bank_y_regs)}."
                    ),
                    level=ValidationLevel.ERROR,
                )
            )

    return errors


def _validate_rdna_isa(node: LogicalNode, level: ValidationLevel) -> list[ValidationError]:
    """Validate AMD RDNA ISA attributes on a node.

    Args:
        node (LogicalNode): RDNA instruction node.
        level (ValidationLevel): Active validation level.

    Returns:
        list[ValidationError]: Detected RDNA validation errors.
    """
    errors: list[ValidationError] = []
    wavefront = node.attributes.get("wavefront_size")
    if wavefront is not None and wavefront not in (32, 64):
        errors.append(
            ValidationError(
                node_id=node.id,
                attribute="wavefront_size",
                message=f"AMD RDNA wavefront_size must be 32 or 64, got {wavefront}.",
                level=ValidationLevel.ERROR,
            )
        )

    reg_classes = node.attributes.get("register_classes")
    if isinstance(reg_classes, dict):
        valid_classes = {"VGPR", "SGPR", "AGPR"}
        for op_name, rclass in reg_classes.items():
            if rclass not in valid_classes:
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute=f"register_classes.{op_name}",
                        message=f"Invalid RDNA register class '{rclass}', expected one of {valid_classes}.",
                        level=ValidationLevel.ERROR,
                    )
                )

    vgpr_indices = node.attributes.get("vgpr_operands")
    if isinstance(vgpr_indices, list) and len(vgpr_indices) >= 2:
        banks = [idx % 4 for idx in vgpr_indices if isinstance(idx, int)]
        if len(banks) != len(set(banks)):
            lvl = ValidationLevel.ERROR if level == ValidationLevel.STRICT else ValidationLevel.WARNING
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="vgpr_operands",
                    message=(
                        f"VGPR bank conflict detected in instruction '{node.op_type}': "
                        f"operands map to same bank ({banks})."
                    ),
                    level=lvl,
                )
            )
    return errors


def _validate_sass_barrier_attributes(node: LogicalNode) -> list[ValidationError]:
    """Validate SASS read_barrier, write_barrier, and wait_barrier_mask attributes.

    Args:
        node (LogicalNode): SASS instruction node.

    Returns:
        list[ValidationError]: Detected barrier mask errors.
    """
    errors: list[ValidationError] = []
    for barrier_key in ("read_barrier", "write_barrier", "wait_barrier_mask"):
        b_val = node.attributes.get(barrier_key)
        if b_val is None:
            continue
        if isinstance(b_val, int) and not isinstance(b_val, bool):
            if not (0 <= b_val <= 63):
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute=barrier_key,
                        message=f"Invalid SASS barrier mask '{b_val}'. Must be in range 0..63.",
                        level=ValidationLevel.ERROR,
                    )
                )
        elif isinstance(b_val, list):
            if not all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= 5 for x in b_val):
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute=barrier_key,
                        message=f"Invalid SASS barrier list '{b_val}'. Entries must be barrier indices 0..5.",
                        level=ValidationLevel.ERROR,
                    )
                )
        elif isinstance(b_val, str):
            if len(b_val) > 6 or not set(b_val).issubset({"0", "1", "2", "3", "4", "5", "-", " "}):
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute=barrier_key,
                        message=f"Invalid SASS barrier mask string '{b_val}'.",
                        level=ValidationLevel.ERROR,
                    )
                )
        else:
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute=barrier_key,
                    message=f"Invalid SASS barrier mask type for '{barrier_key}'.",
                    level=ValidationLevel.ERROR,
                )
            )
    return errors


def _validate_sass_isa(node: LogicalNode) -> list[ValidationError]:
    """Validate NVIDIA SASS ISA attributes on a node.

    Args:
        node (LogicalNode): SASS instruction node.

    Returns:
        list[ValidationError]: Detected SASS validation errors.
    """
    errors: list[ValidationError] = []
    barrier = node.attributes.get("barrier_predicate")
    if (
        barrier is not None
        and isinstance(barrier, str)
        and not (barrier.startswith(("@P", "@!P")) or barrier in ("@PT", "@!PT"))
    ):
        errors.append(
            ValidationError(
                node_id=node.id,
                attribute="barrier_predicate",
                message=f"Invalid SASS barrier predicate '{barrier}'.",
                level=ValidationLevel.ERROR,
            )
        )

    sync = node.attributes.get("warp_sync")
    if sync is not None and sync not in ("sync", "yield", "diverge", "arrive"):
        errors.append(
            ValidationError(
                node_id=node.id,
                attribute="warp_sync",
                message=f"Invalid warp synchronization marker '{sync}'.",
                level=ValidationLevel.ERROR,
            )
        )

    mem_space = node.attributes.get("memory_space")
    valid_spaces = {"global", "shared", "constant", "local"}
    if mem_space is not None and mem_space not in valid_spaces:
        errors.append(
            ValidationError(
                node_id=node.id,
                attribute="memory_space",
                message=f"Invalid SASS memory space '{mem_space}', expected one of {valid_spaces}.",
                level=ValidationLevel.ERROR,
            )
        )

    stall = node.attributes.get("stall_count") if "stall_count" in node.attributes else node.attributes.get("stall")
    if stall is not None and (not isinstance(stall, int) or isinstance(stall, bool) or not (0 <= stall <= 15)):
        errors.append(
            ValidationError(
                node_id=node.id,
                attribute="stall_count",
                message=f"SASS instruction stall count must be an integer between 0 and 15, got {stall}.",
                level=ValidationLevel.ERROR,
            )
        )

    yield_val = node.attributes.get("yield_flag") if "yield_flag" in node.attributes else node.attributes.get("yield")
    if isinstance(yield_val, str) and yield_val not in ("Y", "-", "yield", "noyield"):
        errors.append(
            ValidationError(
                node_id=node.id,
                attribute="yield_flag",
                message=f"Invalid SASS yield flag '{yield_val}'. Expected 'Y' or '-'.",
                level=ValidationLevel.ERROR,
            )
        )
    elif yield_val is not None and not isinstance(yield_val, (str, bool)):
        errors.append(
            ValidationError(
                node_id=node.id,
                attribute="yield_flag",
                message=f"Invalid SASS yield flag '{yield_val}'.",
                level=ValidationLevel.ERROR,
            )
        )

    errors.extend(_validate_sass_barrier_attributes(node))
    return errors


def _validate_wgsl_isa(node: LogicalNode) -> list[ValidationError]:
    """Validate WebGPU WGSL attributes, alignment, and address spaces on a node.

    Args:
        node (LogicalNode): WGSL operation node.

    Returns:
        list[ValidationError]: Detected WGSL validation errors.
    """
    errors: list[ValidationError] = []
    wg_size = node.attributes.get("workgroup_size")
    if wg_size is not None and (
        not isinstance(wg_size, (list, tuple))
        or len(wg_size) < 1
        or len(wg_size) > 3
        or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in wg_size)
    ):
        errors.append(
            ValidationError(
                node_id=node.id,
                attribute="workgroup_size",
                message=f"WGSL @workgroup_size must be 1 to 3 positive integers, got {wg_size}.",
                level=ValidationLevel.ERROR,
            )
        )

    qualifier = node.attributes.get("address_space")
    if qualifier is not None:
        valid_qualifiers = {
            "uniform",
            "storage, read",
            "storage, read_write",
            "workgroup",
            "private",
            "function",
        }
        if qualifier not in valid_qualifiers:
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="address_space",
                    message=f"Invalid WGSL address space qualifier '{qualifier}', expected one of {valid_qualifiers}.",
                    level=ValidationLevel.ERROR,
                )
            )

    if qualifier == "uniform":
        align = (
            node.attributes.get("struct_alignment")
            if "struct_alignment" in node.attributes
            else node.attributes.get("alignment")
        )
        if align is not None and (not isinstance(align, int) or isinstance(align, bool) or align % 16 != 0):
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="struct_alignment",
                    message=f"WGSL uniform buffer struct alignment must be a multiple of 16 bytes, got {align}.",
                    level=ValidationLevel.ERROR,
                )
            )

        stride = node.attributes.get("array_stride")
        if stride is not None and (not isinstance(stride, int) or isinstance(stride, bool) or stride % 16 != 0):
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="array_stride",
                    message=f"WGSL uniform buffer array stride must be a multiple of 16 bytes, got {stride}.",
                    level=ValidationLevel.ERROR,
                )
            )

    builtin = node.attributes.get("builtin")
    if builtin is not None and builtin not in WGSL_COMPUTE_BUILTINS:
        errors.append(
            ValidationError(
                node_id=node.id,
                attribute="builtin",
                message=f"Invalid WGSL compute builtin '{builtin}', expected one of {sorted(WGSL_COMPUTE_BUILTINS)}.",
                level=ValidationLevel.ERROR,
            )
        )

    if qualifier in ("storage, read", "storage_read") and (
        node.op_type in WGSL_MUTATING_OPS or bool(node.attributes.get("is_mutating", False))
    ):
        errors.append(
            ValidationError(
                node_id=node.id,
                attribute="address_space",
                message=(
                    f"Mutating operation '{node.op_type}' is illegal on read-only "
                    f"storage buffer with address space '{qualifier}'."
                ),
                level=ValidationLevel.ERROR,
            )
        )
    return errors


def validate_isa_instruction_node(
    node: LogicalNode, level: ValidationLevel = ValidationLevel.WARNING
) -> list[ValidationError]:
    """Validate GPU accelerator ISA instructions (AMD RDNA, NVIDIA SASS, and WebGPU WGSL).

    Args:
        node (LogicalNode): The node representing an ISA instruction or WGSL primitive.
        level (ValidationLevel): Active validation severity level.

    Returns:
        list[ValidationError]: Detected validation errors or warnings.
    """
    if node.domain in ("amd_rdna", "rdna"):
        return _validate_rdna_isa(node, level)
    if node.domain in ("nvidia_sass", "sass"):
        return _validate_sass_isa(node)
    if node.domain in ("webgpu_wgsl", "wgsl"):
        return _validate_wgsl_isa(node)
    return []


def _parse_wait_barrier_mask(node: LogicalNode) -> set[int]:
    """Extract cleared scoreboard barrier indices from a SASS node's ``wait_barrier_mask``.

    Args:
        node (LogicalNode): SASS instruction node.

    Returns:
        set[int]: Set of cleared barrier indices ``0..5``.
    """
    wait_mask = node.attributes.get("wait_barrier_mask")
    cleared: set[int] = set()
    if isinstance(wait_mask, int) and not isinstance(wait_mask, bool):
        for b_idx in range(6):
            if (wait_mask >> b_idx) & 1:
                cleared.add(b_idx)
    elif isinstance(wait_mask, list):
        cleared.update(b for b in wait_mask if isinstance(b, int) and not isinstance(b, bool))
    elif isinstance(wait_mask, str):
        cleared.update(int(ch) for ch in wait_mask if ch in {"0", "1", "2", "3", "4", "5"})
    return cleared


def validate_sass_scoreboarding_sequence(
    nodes: Sequence[LogicalNode],
    level: ValidationLevel = ValidationLevel.WARNING,
) -> list[ValidationError]:
    """Validate NVIDIA SASS scoreboard dependency latencies between consecutive instructions.

    Args:
        nodes (Sequence[LogicalNode]): Sequential list of SASS instructions to audit.
        level (ValidationLevel): Active validation severity level.

    Returns:
        list[ValidationError]: Detected scoreboard hazards or dependency violations.
    """
    errors: list[ValidationError] = []
    pending_writes: dict[str, tuple[str, str, int, int | None]] = {}
    lvl = ValidationLevel.ERROR if level == ValidationLevel.STRICT else ValidationLevel.WARNING

    for node in nodes:
        if node.domain not in ("nvidia_sass", "sass"):
            continue

        cleared_barriers = _parse_wait_barrier_mask(node)
        if cleared_barriers:
            pending_writes = {r: info for r, info in pending_writes.items() if info[3] not in cleared_barriers}

        for reg in _extract_sass_registers(node, role="src"):
            if reg in pending_writes:
                prod_id, prod_op, rem_cycles, _ = pending_writes[reg]
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="scoreboarding",
                        message=(
                            f"Scoreboard dependency hazard: consumer '{node.id}' ({node.op_type}) "
                            f"reads register '{reg}' produced by '{prod_id}' ({prod_op}) before "
                            f"pipeline latency satisfied ({rem_cycles} stall cycles remaining) "
                            "without barrier synchronization."
                        ),
                        level=lvl,
                    )
                )

        producer_latency = SASS_PIPELINE_LATENCIES.get(node.op_type, 4)
        wb_attr = node.attributes.get("write_barrier")
        wb_id: int | None = wb_attr if isinstance(wb_attr, int) and not isinstance(wb_attr, bool) else None
        for reg in _extract_sass_registers(node, role="dst"):
            pending_writes[reg] = (
                node.id,
                node.op_type,
                producer_latency,
                wb_id,
            )

        stall = node.attributes.get("stall_count")
        if stall is None:
            stall = node.attributes.get("stall")
        elapsed_cycles = max(1, stall) if isinstance(stall, int) and not isinstance(stall, bool) and stall >= 0 else 1
        pending_writes = {
            r: (p_id, p_op, rem - elapsed_cycles, wb)
            for r, (p_id, p_op, rem, wb) in pending_writes.items()
            if rem - elapsed_cycles > 0
        }

    return errors
