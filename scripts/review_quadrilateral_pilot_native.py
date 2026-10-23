#!/usr/bin/env python3
"""Native localhost-free UI for reviewing private quadrilateral pilot samples."""

from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Any, cast

from PIL import Image, ImageOps, ImageTk

from anpr_engine.detection.quadrilateral_review import (
    PilotReviewDecision,
    append_review_event,
    latest_review_state,
    load_pilot_queue,
    load_review_events,
)

CANVAS_WIDTH = 1120
CANVAS_HEIGHT = 500
POINT_RADIUS = 7


def display_window(record: dict[str, Any]) -> tuple[float, float, float, float]:
    center_x, center_y, width, height = record["box_xywh_normalized"]
    pad_x, pad_y = width * 0.20, height * 0.35
    return (
        max(0.0, center_x - width / 2 - pad_x),
        max(0.0, center_y - height / 2 - pad_y),
        min(1.0, center_x + width / 2 + pad_x),
        min(1.0, center_y + height / 2 + pad_y),
    )


class PilotReviewApp:
    def __init__(
        self, root: tk.Tk, *, queue_path: Path, dataset_root: Path, events_path: Path
    ) -> None:
        self.root = root
        self.dataset_root = dataset_root.resolve()
        self.events_path = events_path
        self.queue = load_pilot_queue(queue_path)
        self.records = {record["sample_id"]: record for record in self.queue["records"]}
        self.current: dict[str, Any] | None = None
        self.points: list[tuple[float, float]] = []
        self.photo: ImageTk.PhotoImage | None = None
        self.crop_window = (0.0, 0.0, 1.0, 1.0)
        self.draw_left = 0.0
        self.draw_top = 0.0
        self.draw_width = 1.0
        self.draw_height = 1.0

        root.title("Plaka Köşe İnceleme — Quadrilateral Pilot")
        root.geometry("1200x720")
        root.configure(bg="#111111")
        root.protocol("WM_DELETE_WINDOW", root.destroy)
        root.bind("a", lambda _event: self._approve())
        root.bind("u", lambda _event: self._unusable())
        root.bind("r", lambda _event: self._reset())
        root.bind("<Escape>", lambda _event: self._reset())
        root.bind("<Return>", lambda _event: self._correct())
        root.bind("<Command-z>", lambda _event: self._undo())

        self.status = tk.Label(root, fg="white", bg="#111111", font=("Arial", 16, "bold"))
        self.status.pack(pady=(12, 4))
        self.help = tk.Label(
            root,
            text=(
                "Plaka yüzeyi: mavi bant dahil; siyah çerçeve ve araç gövdesi hariç. "
                "Düzeltme sırası: TL → TR → BR → BL · "
                "A: onayla, U: kullanılamaz, Enter: kaydet, ⌘Z: geri al"
            ),
            fg="#dddddd",
            bg="#111111",
            font=("Arial", 12),
        )
        self.help.pack(pady=(0, 8))
        self.canvas = tk.Canvas(
            root,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg="#222222",
            highlightthickness=1,
            highlightbackground="#555555",
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._click_canvas)

        controls = tk.Frame(root, bg="#111111")
        controls.pack(pady=10)
        self.approve = tk.Button(
            controls, text="Öneriyi onayla", command=self._approve, padx=14, pady=8
        )
        self.approve.pack(side=tk.LEFT, padx=5)
        tk.Button(controls, text="Çizimi kaydet", command=self._correct, padx=14, pady=8).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(controls, text="Kullanılamaz", command=self._unusable, padx=14, pady=8).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(controls, text="Noktaları sıfırla", command=self._reset, padx=14, pady=8).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(controls, text="Son kararı geri al", command=self._undo, padx=14, pady=8).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(controls, text="Şimdilik kapat", command=root.destroy, padx=14, pady=8).pack(
            side=tk.LEFT, padx=5
        )
        self.sample_label = tk.Label(root, fg="#aaaaaa", bg="#111111", font=("Menlo", 10))
        self.sample_label.pack()
        self._load_next()

    def _pending(self) -> list[dict[str, Any]]:
        state = latest_review_state(load_review_events(self.events_path))
        return [
            record
            for identifier, record in self.records.items()
            if identifier not in state or state[identifier].decision is PilotReviewDecision.PENDING
        ]

    def _load_next(self) -> None:
        pending = self._pending()
        reviewed = len(self.records) - len(pending)
        if not pending:
            self.current = None
            self.canvas.delete("all")
            self.status.configure(text=f"Tamamlandı: {reviewed}/{len(self.records)}")
            messagebox.showinfo("Review tamamlandı", "Bütün pilot kayıtları incelendi.")
            return
        self.current = pending[0]
        self.points = []
        self.status.configure(
            text=(
                f"{reviewed}/{len(self.records)} tamamlandı · "
                f"{self.current['split']}/{self.current['cohort']}"
            )
        )
        self.sample_label.configure(text=self.current["sample_id"])
        self.approve.configure(
            state=tk.NORMAL if self.current.get("proposal") is not None else tk.DISABLED
        )
        self._render()

    def _source_path(self) -> Path:
        assert self.current is not None
        path = (self.dataset_root / cast(str, self.current["canonical_image_path"])).resolve()
        if self.dataset_root not in path.parents or not path.is_file():
            raise ValueError("sample image escapes dataset root or is absent")
        return path

    def _render(self) -> None:
        assert self.current is not None
        with Image.open(self._source_path()) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        width, height = image.size
        self.crop_window = display_window(self.current)
        left, top, right, bottom = self.crop_window
        crop = image.crop(
            (
                round(left * width),
                round(top * height),
                round(right * width),
                round(bottom * height),
            )
        )
        scale = min(CANVAS_WIDTH / crop.width, CANVAS_HEIGHT / crop.height)
        output_size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
        resized = crop.resize(output_size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.draw_width, self.draw_height = output_size
        self.draw_left = (CANVAS_WIDTH - self.draw_width) / 2
        self.draw_top = (CANVAS_HEIGHT - self.draw_height) / 2
        self.canvas.delete("all")
        self.canvas.create_image(self.draw_left, self.draw_top, image=self.photo, anchor=tk.NW)
        self._draw_overlays()

    def _to_canvas(self, point: tuple[float, float]) -> tuple[float, float]:
        left, top, right, bottom = self.crop_window
        return (
            self.draw_left + (point[0] - left) / (right - left) * self.draw_width,
            self.draw_top + (point[1] - top) / (bottom - top) * self.draw_height,
        )

    def _to_source(self, x: float, y: float) -> tuple[float, float]:
        left, top, right, bottom = self.crop_window
        local_x = (x - self.draw_left) / self.draw_width
        local_y = (y - self.draw_top) / self.draw_height
        return left + local_x * (right - left), top + local_y * (bottom - top)

    def _draw_overlays(self) -> None:
        assert self.current is not None
        center_x, center_y, width, height = self.current["box_xywh_normalized"]
        box_left, box_top = self._to_canvas((center_x - width / 2, center_y - height / 2))
        box_right, box_bottom = self._to_canvas((center_x + width / 2, center_y + height / 2))
        self.canvas.create_rectangle(
            box_left,
            box_top,
            box_right,
            box_bottom,
            outline="#ffd54f",
            width=2,
            dash=(7, 5),
        )
        shown: list[tuple[float, float]] = self.points
        if not shown and self.current.get("proposal") is not None:
            shown = [tuple(point) for point in self.current["proposal"]["corners_xy_normalized"]]
        if shown:
            canvas_points = [self._to_canvas(point) for point in shown]
            if len(canvas_points) == 4:
                flat = [value for point in canvas_points for value in point]
                self.canvas.create_polygon(*flat, outline="#00e676", fill="", width=3)
            for index, (x, y) in enumerate(canvas_points):
                self.canvas.create_oval(
                    x - POINT_RADIUS,
                    y - POINT_RADIUS,
                    x + POINT_RADIUS,
                    y + POINT_RADIUS,
                    fill="#ff5252",
                    outline="white",
                    width=2,
                )
                self.canvas.create_text(
                    x,
                    y - 16,
                    text=("TL", "TR", "BR", "BL")[index],
                    fill="white",
                    font=("Arial", 11, "bold"),
                )

    def _click_canvas(self, event: tk.Event[tk.Misc]) -> None:
        if self.current is None:
            return
        if not (
            self.draw_left <= event.x <= self.draw_left + self.draw_width
            and self.draw_top <= event.y <= self.draw_top + self.draw_height
        ):
            return
        if len(self.points) == 4:
            self.points = []
        self.points.append(self._to_source(float(event.x), float(event.y)))
        self._render()

    def _save(
        self,
        decision: PilotReviewDecision,
        corners: list[dict[str, float]] | None,
        reason: str,
    ) -> None:
        assert self.current is not None
        try:
            append_review_event(
                self.events_path,
                sample_id=self.current["sample_id"],
                image_sha256=self.current["image_sha256"],
                label_sha256=self.current["label_sha256"],
                decision=decision,
                corners=corners,
                reason=reason,
            )
        except ValueError as error:
            messagebox.showerror("Geçersiz dörtgen", str(error))
            return
        self._load_next()

    def _approve(self) -> None:
        assert self.current is not None
        proposal = self.current.get("proposal")
        if proposal is None:
            return
        corners = [{"x": point[0], "y": point[1]} for point in proposal["corners_xy_normalized"]]
        self._save(
            PilotReviewDecision.APPROVED_PROPOSAL,
            corners,
            "HUMAN_APPROVED_PROPOSAL",
        )

    def _correct(self) -> None:
        if len(self.points) != 4:
            messagebox.showwarning("Dört nokta gerekli", "TL → TR → BR → BL sırasıyla tıkla.")
            return
        corners = [{"x": x, "y": y} for x, y in self.points]
        self._save(PilotReviewDecision.CORRECTED, corners, "HUMAN_CORRECTED_CORNERS")

    def _unusable(self) -> None:
        self._save(PilotReviewDecision.UNUSABLE, None, "HUMAN_MARKED_UNUSABLE")

    def _reset(self) -> None:
        self.points = []
        self._render()

    def _undo(self) -> None:
        events = load_review_events(self.events_path)
        latest_non_pending = next(
            (
                event
                for event in reversed(events)
                if event.decision is not PilotReviewDecision.PENDING
            ),
            None,
        )
        if latest_non_pending is None:
            messagebox.showinfo(
                "Geri alınacak karar yok", "Henüz kaydedilmiş bir karar bulunmuyor."
            )
            return
        record = self.records[latest_non_pending.sample_id]
        append_review_event(
            self.events_path,
            sample_id=latest_non_pending.sample_id,
            image_sha256=record["image_sha256"],
            label_sha256=record["label_sha256"],
            decision=PilotReviewDecision.PENDING,
            corners=None,
            reason="HUMAN_UNDO_LAST_DECISION",
        )
        self._load_next()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    args = parser.parse_args()
    root = tk.Tk()
    PilotReviewApp(
        root,
        queue_path=args.queue,
        dataset_root=args.dataset_root,
        events_path=args.events,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
