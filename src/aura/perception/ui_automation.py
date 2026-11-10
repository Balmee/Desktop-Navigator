
from pywinauto import Desktop


MAX_ELEMENTS = 120


def _safe_property(obj, property_name, default=None):
    try:
        value = getattr(obj, property_name)
        return value if value is not None else default
    except Exception:
        return default


class UIAutomationInspector:

    def inspect_window(self, hwnd: int) -> dict:
        """
        Inspect the UI Automation tree of one window.

        This is read-only. It does not click controls,
        type text or retrieve editable field contents.
        """

        if not hwnd:
            raise ValueError("Invalid window handle.")

        window = Desktop(
            backend="uia"
        ).window(handle=hwnd)

        elements = []
        skipped_password_fields = 0

        try:
            descendants = window.descendants()
        except Exception as error:
            return {
                "success": False,
                "error": str(error),
                "elements": [],
                "element_count": 0,
            }

        for wrapper in descendants[:MAX_ELEMENTS]:
            info = wrapper.element_info

            control_type = _safe_property(
                info, "control_type", "Unknown"
            )

            is_password = bool(
                _safe_property(info, "is_password", False)
            )

            # Never retain metadata from password controls.
            if is_password:
                skipped_password_fields += 1
                continue

            try:
                rect = wrapper.rectangle()

                bounds = {
                    "left": rect.left,
                    "top": rect.top,
                    "right": rect.right,
                    "bottom": rect.bottom,
                    "width": rect.width(),
                    "height": rect.height(),
                }
            except Exception:
                bounds = None

            element = {
                "name": _safe_property(info, "name", ""),
                "control_type": control_type,
                "automation_id": _safe_property(
                    info, "automation_id", ""
                ),
                "class_name": _safe_property(
                    info, "class_name", ""
                ),
                "enabled": _safe_property(
                    info, "enabled", None
                ),
                "visible": _safe_property(
                    info, "visible", None
                ),
                "bounds": bounds,
            }

            elements.append(element)

        return {
            "success": True,
            "elements": elements,
            "element_count": len(elements),
            "truncated": len(descendants) > MAX_ELEMENTS,
            "skipped_password_fields": skipped_password_fields,
        }
