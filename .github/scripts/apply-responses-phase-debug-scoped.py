from pathlib import Path


driver_path = Path("../patch/.github/scripts/apply-responses-phase-debug.py")
source = driver_path.read_text()

old_helper = '''def replace_once(text: str, old: str, new: str, label: str) -> str:\n    count = text.count(old)\n    if count != 1:\n        raise SystemExit(f"{label}: expected exactly one match, found {count}")\n    return text.replace(old, new, 1)\n'''

new_helper = '''def replace_once(text: str, old: str, new: str, label: str) -> str:\n    count = text.count(old)\n    if label == "disable lazy grammar for Responses phases" and count > 1:\n        marker = "static common_chat_params common_chat_params_init_qwen3_coder"\n        pos = text.find(marker)\n        if pos == -1:\n            raise SystemExit(f"{label}: Qwen parser marker not found")\n        prefix = text[:pos]\n        suffix = text[pos:]\n        suffix_count = suffix.count(old)\n        if suffix_count < 1:\n            raise SystemExit(f"{label}: target not found after Qwen parser marker")\n        return prefix + suffix.replace(old, new, 1)\n    if count != 1:\n        raise SystemExit(f"{label}: expected exactly one match, found {count}")\n    return text.replace(old, new, 1)\n'''

count = source.count(old_helper)
if count != 1:
    raise SystemExit(f"driver helper: expected exactly one match, found {count}")

source = source.replace(old_helper, new_helper, 1)
exec(compile(source, str(driver_path), "exec"), {"__name__": "__main__"})
