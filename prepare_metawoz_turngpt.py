#!/usr/bin/env python3
"""Prepare MetaLWOZ train/validation and official heldout test dialogues."""
from __future__ import annotations
import argparse, hashlib, io, json, re, unicodedata, zipfile
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from turngpt_common import SPEAKER1, SPEAKER2, SPECIAL_TOKENS

ROOT=Path(__file__).parent
RAW=ROOT/"data/raw/metawoz"
SPLIT=ROOT/"data/splits/metawoz_splits.json"

def norm(s):
    s=re.sub(r"[,.:;]+(\w+)",r" \1",s)
    s=unicodedata.normalize("NFD",s).lower()
    s="".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s\s+"," ",re.sub(r'[.,!?:;()\[\]"-]',"",s)).strip()

def read_train(z):
    out=[]
    for name in sorted(n for n in z.namelist() if n.startswith("dialogues/") and n.endswith(".txt")):
        for i,line in enumerate(z.read(name).decode().splitlines()):
            if line.strip():
                d=json.loads(line); out.append((f"{name}:{i}",d["turns"]))
    return out

def read_test(z):
    nested=zipfile.ZipFile(io.BytesIO(z.read("dstc8_metalwoz_heldout.zip")))
    out=[]
    for name in sorted(n for n in nested.namelist() if n.startswith("dialogues/") and n.endswith(".txt")):
        for i,line in enumerate(nested.read(name).decode().splitlines()):
            if not line.strip():
                continue
            try:
                d=json.loads(line)
            except json.JSONDecodeError:
                continue
            turns=d.get("turns")
            if isinstance(turns,list) and turns and all(isinstance(x,str) for x in turns):
                out.append((f"heldout:{name}:{i}:{d.get('id','')}",turns))
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--max-length",type=int,default=256); ap.add_argument("--keep-length",type=int,default=64); ap.add_argument("--overlap-length",type=int,default=10); args=ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(args.output_dir)
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True); tok.pad_token=tok.pad_token or tok.eos_token; tok.add_special_tokens(SPECIAL_TOKENS)
    ids={n:tok.convert_tokens_to_ids(t) for n,t in (("ts","<ts>"),("speaker1",SPEAKER1),("speaker2",SPEAKER2))}
    with zipfile.ZipFile(RAW/"metalwoz-v1.zip") as z: train=read_train(z)
    with zipfile.ZipFile(RAW/"metalwoz-test-v1.zip") as z: test=read_test(z)
    split_idx=json.loads(SPLIT.read_text()); records={"train":[train[i] for i in split_idx["train"]],"validation":[train[i] for i in split_idx["validation"]],"test":test}
    args.output_dir.mkdir(parents=True); handles={s:(args.output_dir/f"{s}.jsonl").open("w") for s in records}; reports={s:Counter() for s in records}
    try:
      for split,items in records.items():
        texts=[]; good=[]
        for sid,turns in items:
          reports[split]["source_dialogues"]+=1
          turns=[str(t).strip() for t in turns if str(t).strip()]
          if len(turns)<2 or any(not re.search(r"\w",t) for t in turns): reports[split]["filtered_dialogues"]+=1; continue
          texts.append(norm(turns[0])+"".join("<ts> "+norm(t) for t in turns[1:])+"<ts>"); good.append((sid,turns))
        enc=tok(texts,add_special_tokens=False,truncation=False)["input_ids"] if texts else []
        for (sid,turns),input_ids in zip(good,enc):
          sp=[]; state=ids["speaker1"]
          for x in input_ids: sp.append(state); state=ids["speaker2"] if x==ids["ts"] and state==ids["speaker1"] else ids["speaker1"] if x==ids["ts"] else state
          reports[split]["accepted_dialogues"]+=1; reports[split]["turns"]+=len(turns); reports[split]["tokens"]+=len(input_ids); reports[split]["ts_tokens"]+=input_ids.count(ids["ts"])
          stride=args.max_length-args.overlap_length
          for start in range(0,len(input_ids),stride):
            w=input_ids[start:start+args.max_length]
            if len(w)<args.keep_length: break
            handles[split].write(json.dumps({"dataset":"metawoz","source_split":split,"source_id":sid,"start_token":start,"input_ids":w,"speaker_ids":sp[start:start+args.max_length]},separators=(",",":"))+"\n"); reports[split]["windows"]+=1
    finally:
      [h.close() for h in handles.values()]
    manifest={"status":"ok","scope":"metawoz","raw_sha256":{"train":hashlib.sha256((RAW/"metalwoz-v1.zip").read_bytes()).hexdigest(),"test":hashlib.sha256((RAW/"metalwoz-test-v1.zip").read_bytes()).hexdigest()},"settings":{"max_length":args.max_length,"keep_length":args.keep_length,"overlap_length":args.overlap_length},"special_token_ids":ids,"splits":{s:dict(v) for s,v in reports.items()}}
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n"); print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
