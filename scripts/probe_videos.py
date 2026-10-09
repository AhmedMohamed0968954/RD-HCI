"""Check every video header and first decoded frame; report failures without dropping events."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from epic_starter.data.events import load_events
from epic_starter.data.manifests import video_index


def probe(item):
    import av
    video_id, filename = item
    try:
        with av.open(filename) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            frame = next(container.decode(stream),None)
            if frame is None or frame.is_corrupt:
                raise ValueError("Missing/corrupt first frame")
            duration = float(stream.duration*stream.time_base) if stream.duration is not None else None
            return dict(video_id=video_id,path=filename,status="ok",width=frame.width,
                        height=frame.height,duration_seconds=duration)
    except Exception as exc:
        return dict(video_id=video_id,path=filename,status="failed",error=str(exc))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root",required=True)
    parser.add_argument("--annotation-csv",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--workers",type=int,default=1)
    args=parser.parse_args()
    if args.workers < 1:
        parser.error("Positive workers required")
    out=Path(args.out)
    if out.exists():
        raise FileExistsError("Choose a new health-report path")
    index=video_index(Path(args.video_root))
    events=load_events(Path(args.annotation_csv))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(probe,[(k,str(v)) for k,v in sorted(index.items())]))
    bad={r["video_id"] for r in results if r["status"]!="ok"}
    missing={e.video_id for e in events}-set(index)
    affected=[e.narration_id for e in events if e.video_id in bad|missing]
    report=dict(check="header and first frame only; not a full decode of every frame",
                videos=len(results),failed_video_ids=sorted(bad),missing_video_ids=sorted(missing),
                affected_events=len(affected),affected_narration_ids=affected,results=results)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in {"results","affected_narration_ids"}},indent=2))
    if bad or missing:
        raise SystemExit(1)


if __name__=="__main__":
    main()
