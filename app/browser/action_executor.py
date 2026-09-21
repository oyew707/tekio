"""Execute normalized-coordinate `computer_use` actions in Playwright."""

from __future__ import annotations

from typing import Any


class ActionExecutor:
    def __init__(self, session: Any, viewport: tuple[int, int] = (1440, 900)) -> None:
        self.session = session
        self.viewport = viewport

    def _scale_coordinates(self, x: float, y: float) -> tuple[float, float]:
        width, height = self.viewport
        scaled_x = max(0.0, min(width, (x / 1000.0) * width))
        scaled_y = max(0.0, min(height, (y / 1000.0) * height))
        return scaled_x, scaled_y

    async def execute(self, action: dict[str, Any]) -> str:
        kind = action.get("action")
        page = self.session.page

        if kind == "goto":
            await page.goto(action["url"])
        elif kind in {"click", "move"}:
            x, y = self._scale_coordinates(float(action["x"]), float(action["y"]))
            await page.mouse.move(x, y)
            if kind == "click":
                await page.mouse.click(x, y)
        elif kind == "type":
            await page.keyboard.type(action.get("text", ""))
        elif kind == "press":
            await page.keyboard.press(action["key"])
        elif kind == "scroll":
            delta_y = int(action.get("delta_y", 0))
            await page.mouse.wheel(0, delta_y)
        else:
            raise ValueError(f"Unsupported computer_use action: {kind}")

        return await self.session.screenshot()
