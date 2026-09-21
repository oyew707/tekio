"""Sandboxed Playwright browser session utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright


class PlaywrightSession:
    def __init__(self, viewport: dict[str, int] | None = None, headless: bool = True) -> None:
        self.viewport = viewport or {"width": 1440, "height": 900}
        self.headless = headless
        self._playwright: Any = None
        self.browser: Any = None
        self.context: Any = None
        self.page: Any = None

    async def start(self) -> None:
        self._playwright = await async_playwright().start()
        self.browser = await self._playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context(viewport=self.viewport)
        self.page = await self.context.new_page()

    async def stop(self) -> None:
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def screenshot(self, output_path: str = "artifacts/latest.png") -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        await self.page.screenshot(path=output_path)
        return output_path
