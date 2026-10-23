#!/usr/bin/env python3
# ruff: noqa: E501
"""Serve a localhost-only corner review UI for a private pilot queue."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Any, cast

import uvicorn
from PIL import Image, ImageOps
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from anpr_engine.detection.quadrilateral_review import (
    PilotReviewDecision,
    append_review_event,
    latest_review_state,
    load_pilot_queue,
    load_review_events,
)

HTML = """<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Quadrilateral Pilot Review</title>
<style>body{font-family:system-ui;margin:0;background:#111;color:#eee}main{max-width:1200px;margin:auto;padding:20px}
#stage{position:relative;display:inline-block;max-width:100%;background:#222}img{display:block;max-width:100%;max-height:72vh}
svg{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}.box{fill:none;stroke:#ffd54f;stroke-width:2;stroke-dasharray:7 5}
.quad{fill:#00e67622;stroke:#00e676;stroke-width:3}.point{fill:#ff5252;stroke:white;stroke-width:1.5}
button{margin:8px 8px 0 0;padding:10px 14px}code{word-break:break-all}.muted{color:#aaa}</style></head><body><main>
<h1>Quadrilateral pilot review</h1><p id="status"></p><div id="stage"><img id="image"><svg id="overlay" viewBox="0 0 1 1" preserveAspectRatio="none"></svg></div>
<p>TL → TR → BR → BL sırasıyla dört noktaya tıkla. Mevcut yeşil öneriyi onaylayabilir veya yeniden çizebilirsin.</p>
<button id="approve">Öneriyi onayla</button><button id="correct">Çizimi kaydet</button><button id="unusable">Kullanılamaz</button><button id="reset">Noktaları sıfırla</button>
<p class="muted"><code id="sample"></code></p></main><script>
let record=null,points=[];const svg=document.querySelector('#overlay'),img=document.querySelector('#image');
function local(p){const [l,t,r,b]=record.display_window_xyxy;return[(p[0]-l)/(r-l),(p[1]-t)/(b-t)]}
function source(p){const [l,t,r,b]=record.display_window_xyxy;return[l+p[0]*(r-l),t+p[1]*(b-t)]}
function draw(){svg.innerHTML='';if(!record)return;const b=record.box_xywh_normalized,[cx,cy,w,h]=b;
const tl=local([cx-w/2,cy-h/2]),br=local([cx+w/2,cy+h/2]);
svg.innerHTML+=`<rect class="box" x="${tl[0]}" y="${tl[1]}" width="${br[0]-tl[0]}" height="${br[1]-tl[1]}"/>`;
const shown=points.length?points:(record.proposal?.corners_xy_normalized||[]);if(shown.length){
const visible=shown.map(local);svg.innerHTML+=`<polygon class="quad" points="${visible.map(p=>p.join(',')).join(' ')}"/>`;
visible.forEach((p,i)=>svg.innerHTML+=`<circle class="point" cx="${p[0]}" cy="${p[1]}" r=".009"><title>${['TL','TR','BR','BL'][i]}</title></circle>`);}}
async function load(){const r=await fetch('/api/current');if(r.status===204){document.querySelector('#status').textContent='İnceleme tamamlandı.';return}
record=await r.json();points=[];img.src='/crop/'+record.sample_id;document.querySelector('#sample').textContent=record.sample_id;
document.querySelector('#status').textContent=`${record.progress.reviewed}/${record.progress.total} tamamlandı · ${record.split}/${record.cohort}`;img.onload=draw;draw();}
svg.onclick=e=>{if(points.length===4)points=[];const r=svg.getBoundingClientRect();points.push(source([(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height]));draw()};
async function decide(decision,corners,reason){const r=await fetch('/api/decision',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({sample_id:record.sample_id,decision,corners,reason})});if(!r.ok){alert(await r.text());return}await load()}
document.querySelector('#approve').onclick=()=>record.proposal?decide('APPROVED_PROPOSAL',record.proposal.corners_xy_normalized.map(p=>({x:p[0],y:p[1]})),'HUMAN_APPROVED_PROPOSAL'):alert('Onaylanacak öneri yok.');
document.querySelector('#correct').onclick=()=>points.length===4?decide('CORRECTED',points.map(p=>({x:p[0],y:p[1]})),'HUMAN_CORRECTED_CORNERS'):alert('Dört nokta gerekli.');
document.querySelector('#unusable').onclick=()=>decide('UNUSABLE',null,'HUMAN_MARKED_UNUSABLE');document.querySelector('#reset').onclick=()=>{points=[];draw()};load();
</script></body></html>"""


def create_app(queue_path: Path, dataset_root: Path, events_path: Path) -> Starlette:
    queue = load_pilot_queue(queue_path)
    records = {record["sample_id"]: record for record in queue["records"]}

    def display_window(record: dict[str, Any]) -> list[float]:
        center_x, center_y, width, height = record["box_xywh_normalized"]
        pad_x, pad_y = width * 0.20, height * 0.35
        return [
            max(0.0, center_x - width / 2 - pad_x),
            max(0.0, center_y - height / 2 - pad_y),
            min(1.0, center_x + width / 2 + pad_x),
            min(1.0, center_y + height / 2 + pad_y),
        ]

    async def index(_: Request) -> HTMLResponse:
        return HTMLResponse(HTML, headers={"Cache-Control": "no-store"})

    async def current(_: Request) -> JSONResponse:
        state = latest_review_state(load_review_events(events_path))
        pending = [
            record
            for identifier, record in records.items()
            if identifier not in state or state[identifier].decision is PilotReviewDecision.PENDING
        ]
        if not pending:
            return JSONResponse({}, status_code=204)
        record = dict(pending[0])
        record["display_window_xyxy"] = display_window(record)
        record["progress"] = {"reviewed": len(records) - len(pending), "total": len(records)}
        return JSONResponse(record, headers={"Cache-Control": "no-store"})

    async def crop(request: Request) -> Response:
        record = records.get(request.path_params["sample_id"])
        if record is None:
            raise ValueError("unknown sample")
        candidate = (dataset_root / record["canonical_image_path"]).resolve()
        if dataset_root.resolve() not in candidate.parents or not candidate.is_file():
            raise ValueError("sample image escapes the dataset root or is absent")
        if candidate.stat().st_size <= 0:
            raise ValueError("sample image is empty")
        with Image.open(candidate) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            left, top, right, bottom = display_window(record)
            cropped = image.crop(
                (
                    round(left * width),
                    round(top * height),
                    round(right * width),
                    round(bottom * height),
                )
            )
            payload = io.BytesIO()
            cropped.save(payload, format="JPEG", quality=92, optimize=True)
        return Response(
            payload.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "no-store"}
        )

    async def decision(request: Request) -> JSONResponse:
        payload = cast(dict[str, Any], await request.json())
        record = records.get(payload.get("sample_id"))
        if record is None:
            return JSONResponse({"error": "unknown sample"}, status_code=400)
        try:
            value = PilotReviewDecision(str(payload.get("decision", "")))
            event = append_review_event(
                events_path,
                sample_id=record["sample_id"],
                image_sha256=record["image_sha256"],
                label_sha256=record["label_sha256"],
                decision=value,
                corners=payload.get("corners"),
                reason=str(payload.get("reason", "")),
            )
        except (TypeError, ValueError) as error:
            return JSONResponse({"error": str(error)}, status_code=400)
        return JSONResponse(event.model_dump(mode="json"))

    return Starlette(
        routes=[
            Route("/", index),
            Route("/api/current", current),
            Route("/crop/{sample_id}", crop),
            Route("/api/decision", decision, methods=["POST"]),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    uvicorn.run(
        create_app(args.queue, args.dataset_root, args.events), host="127.0.0.1", port=args.port
    )


if __name__ == "__main__":
    main()
