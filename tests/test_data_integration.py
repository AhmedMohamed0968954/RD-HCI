"""Exercise the public Dataset, exporter and inspection entry points end to end."""
import csv
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import av
import numpy as np
import pytest
import torch
from dataset import EpicKitchensDataset, make_dataloader
from epic_starter.data.manifests import prepare, read_manifest
from scripts.check_data import audit_legacy_manifest, validate_batch, verify_splits

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path):
    videos = tmp_path/"videos"
    videos.mkdir()
    source = videos/"P01_01.mp4"
    with av.open(str(source),"w") as out:
        stream = out.add_stream("mpeg4",rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for i in range(31):
            pixels = np.zeros((32,32,3),dtype=np.uint8)
            pixels[:,:,1 if i<=10 else 0] = 240
            frame = av.VideoFrame.from_ndarray(pixels,format="rgb24")
            frame.pts,frame.time_base = i,Fraction(1,10)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    shutil.copyfile(source,videos/"P02_01.MP4")
    rows = [dict(narration_id=f"e{i}",participant_id=f"P0{1+i//2}",
                 video_id=f"P0{1+i//2}_01",start_timestamp=start,stop_timestamp=stop,
                 verb_class=i,noun_class=i+2)
            for i,(start,stop) in enumerate([(.4,.8),(2.,3.),(.5,.9),(2.,3.)])]
    csv_path = tmp_path/"annotations.csv"
    write_rows(csv_path,rows)
    return csv_path,videos,rows


def write_rows(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_direct_ar_sta_batches_and_empty_history(project):
    csv_path,videos,_=project
    for task in ["AR","STA"]:
        ds=EpicKitchensDataset(str(csv_path),video_root=videos,task=task,num_frames=4,image_size=32)
        batch=next(iter(make_dataloader(ds,batch_size=4)))
        validate_batch(batch)
        assert batch["frames"].shape==(4,4,3,32,32)
        assert batch["frames"] is batch["video"]
        assert batch["verb_label"].tolist()==[0,1,2,3]
        if task=="STA":
            assert batch["use_prior"].tolist()==[True,False,True,False]
            assert not batch["frames"][0].any()
            assert batch["frames"][1,:,0].max()<.1  # Never include red future frames.


def test_normalization_and_transform_contract(project):
    csv_path,videos,_=project
    ds=EpicKitchensDataset(str(csv_path),video_root=videos,task="STA",num_frames=4,image_size=32,
                          normalize=True,mean=(.5,.5,.5),std=(.5,.5,.5))
    assert not ds[0]["frames"].any()  # Empty placeholders remain zero after normalization.
    assert ds[1]["frames"].min() >= -1
    bad=EpicKitchensDataset(str(csv_path),video_root=videos,image_size=32,
                           transform=lambda x:x.float()*2)
    with pytest.raises(ValueError,match="unnormalized"):
        bad[0]


def test_prepared_split_context_and_statistics(project,tmp_path):
    csv_path,videos,_=project
    out=tmp_path/"split"
    prepare(csv_path,videos,out,context=.4)
    assert sum(verify_splits(out).values())==8
    counts=json.loads((out/"class_counts.json").read_text())
    assert sum(counts["train"]["verb"])+sum(counts["val"]["verb"])==4
    ds=EpicKitchensDataset(str(out/"train_sta.jsonl"),video_root=videos,image_size=32)
    for sample in ds:
        if not sample["use_prior"]:
            assert sample["frame_times_seconds"].min()>=.6
    with pytest.raises(ValueError,match="Task"):
        EpicKitchensDataset(str(out/"train_ar.jsonl"),video_root=videos,task="STA")
    records=read_manifest(out/"train_sta.jsonl")
    records[0]["observation"]["end"]+=1
    (out/"bad.jsonl").write_text("\n".join(json.dumps(x) for x in records))
    with pytest.raises(ValueError,match="cutoff"):
        read_manifest(out/"bad.jsonl")


def test_legacy_signature_and_reject_sta(project,tmp_path):
    csv_path,videos,rows=project
    row=rows[1].copy()
    row["clip_path"]="videos/P01_01.mp4"
    path=tmp_path/"legacy.csv"
    write_rows(path,[row])
    with pytest.warns(UserWarning):
        data=EpicKitchensDataset(str(path),4,None,image_size=32)
    assert data[0]["video"].shape==(4,3,32,32)
    assert not data[0]["timing_verified"]
    assert torch.isnan(data[0]["frame_times_seconds"]).all()
    with pytest.raises(ValueError,match="STA"):
        EpicKitchensDataset(str(path),task="STA")
    report=audit_legacy_manifest(path,csv_path)
    assert report["covered_events"]==1 and report["missing_annotation_events"]==3


def test_unlabeled_inference_and_missing_video(project,tmp_path):
    csv_path,videos,rows=project
    for row in rows:
        row.pop("verb_class");row.pop("noun_class")
    path=tmp_path/"test.csv"
    write_rows(path,rows)
    ds=EpicKitchensDataset(str(path),video_root=videos,image_size=32)
    assert not ds[0]["has_label"] and ds[0]["verb_label"]==-1
    (videos/"P02_01.MP4").unlink()
    with pytest.raises(FileNotFoundError):
        EpicKitchensDataset(str(path),video_root=videos)


def test_multiworker_no_drop_and_repeatable_shuffle(project):
    csv_path,videos,_=project
    ds=EpicKitchensDataset(str(csv_path),video_root=videos,image_size=32,num_frames=2)
    def collect(workers):
        return [eid for batch in make_dataloader(ds,3,workers,shuffle=True,seed=7)
                for eid in batch["narration_id"]]
    assert collect(0)==collect(2)
    assert len(set(collect(0)))==4


def test_export_failure_is_retained(project,tmp_path):
    csv_path,videos,rows=project
    # Missing source must fail visibly even without ffmpeg installed.
    rows[0]["video_id"]="missing"
    broken=tmp_path/"broken.csv";write_rows(broken,[rows[0]])
    manifest=tmp_path/"failed.csv"
    result=subprocess.run([sys.executable,str(ROOT/"extract_clips.py"),"--csv_path",str(broken),
                           "--video_dir",str(videos),"--output_dir",str(tmp_path/"clips"),
                           "--output_csv",str(manifest)],capture_output=True,text=True)
    assert result.returncode==1,result.stderr
    with manifest.open() as f:
        saved=list(csv.DictReader(f))
    assert len(saved)==1 and saved[0]["extraction_status"]=="failed"
    assert "not found" in saved[0]["extraction_error"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None,reason="ffmpeg unavailable")
def test_real_export_and_reuse(project,tmp_path):
    csv_path,videos,_=project
    manifest=tmp_path/"clips.csv"
    args=[sys.executable,str(ROOT/"extract_clips.py"),"--csv_path",str(csv_path),
          "--video_dir",str(videos),"--output_dir",str(tmp_path/"clips"),
          "--output_csv",str(manifest),"--limit","2"]
    subprocess.run(args,check=True,capture_output=True)
    with pytest.warns(UserWarning):
        ds=EpicKitchensDataset(str(manifest),4,image_size=32)
    assert len(ds)==2 and ds[0]["frames"].shape==(4,3,32,32)
    from extract_clips import trim_segment
    with csv_path.open() as f:
        row=next(csv.DictReader(f))
    _,clip,error=trim_segment(row,str(videos),str(tmp_path/"clips"),False)
    assert clip and error is None


def test_check_entrypoint(project,tmp_path):
    csv_path,videos,_=project
    manifests=tmp_path/"prepared"
    prepare(csv_path,videos,manifests)
    out=tmp_path/"inspection"
    subprocess.run([sys.executable,str(ROOT/"scripts/check_data.py"),"--manifest-dir",str(manifests),
                    "--video-root",str(videos),"--out-dir",str(out),"--samples","2",
                    "--workers","0","--image-size","32","--num-frames","2"],check=True,capture_output=True)
    report=json.loads((out/"report.json").read_text())
    assert len(report["checks"])==4
    assert len(list(out.glob("*.png")))==8


def test_video_health_reports_corruption_without_losing_events(project,tmp_path):
    csv_path,videos,_=project
    (videos/"P02_01.MP4").write_bytes(b"not an mp4")
    out=tmp_path/"health.json"
    result=subprocess.run([sys.executable,str(ROOT/"scripts/probe_videos.py"),
                           "--video-root",str(videos),"--annotation-csv",str(csv_path),
                           "--out",str(out)],capture_output=True,text=True)
    assert result.returncode==1,result.stderr
    report=json.loads(out.read_text())
    assert report["failed_video_ids"]==["P02_01"]
    assert report["affected_events"]==2
    manifests=tmp_path/"all-events"
    summary=prepare(csv_path,videos,manifests)
    assert sum(s["events"] for s in summary["splits"].values())==4
