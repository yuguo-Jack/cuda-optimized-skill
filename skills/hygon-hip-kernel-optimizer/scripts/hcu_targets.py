"""User-confirmed HCU names, not instruction/toolchain capability declarations."""
import re

HCU_NAMES = {
    "gfx926": "孔明", "gfx928": "孔明e",
    "gfx936": "伯温a0（北美洲）", "gfx938": "伯温b0（南美洲）",
    "gfx946": "少伯", "gfx948": "塞班b1", "gfx92a": "月英",
}


def normalize_gfx(value):
    arch = str(value or "").split(":", 1)[0].lower()
    return arch if re.fullmatch(r"gfx[0-9a-f]+", arch) else None


def describe(value):
    arch = normalize_gfx(value)
    return {"gfx": arch, "architecture_name": HCU_NAMES.get(arch),
            "capabilities_verified": False}
