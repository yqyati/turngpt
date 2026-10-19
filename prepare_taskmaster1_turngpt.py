#!/usr/bin/env python3
"""Prepare official Taskmaster-1 one-person dialogs using train/dev/test CSV IDs."""
from __future__ import annotations
import argparse, csv, json, re, unicodedata
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from turngpt_common import SPEAKER1, SPEAKER2, SPECIAL_TOKENS

ROOT=Path(__file__).parent
RAW=ROOT/"data/raw/taskmaster/taskmaster1"

def norm(s):
    s=re.sub(r"[,.:;]+(\w+)",r" \1",s); s=unicodedata.normalize("NFD",s).lower()
    s="".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s\s+"," ",re.sub(r'[.,!?:;()\[\]"-]',"",s)).strip()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--max-length",type=int,default=256); ap.add_argument("--keep-length",type=int,default=64); ap.add_argument("--overlap-length",type=int,default=10); args=ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(args.output_dir)
    ids_by_split={}
    for split,file in (("train","train.csv"),("validation","dev.csv"),("test","test.csv")):
        with (RAW/file).open() as f: ids_by_split[split]={row[0] for row in csv.reader(f) if row and row[0]}
    data=json.loads((RAW/"self-dialogs.json").read_text()); by_id={x.get("conversation_id"):x for x in data}
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True); tok.pad_token=tok.pad_token or tok.eos_token; tok.add_special_tokens(SPECIAL_TOKENS)
    ids={n:tok.convert_tokens_to_ids(t) for n,t in (("ts","<ts>"),("speaker1",SPEAKER1),("speaker2",SPEAKER2))}
    args.output_dir.mkdir(parents=True); handles={s:(args.output_dir/f"{s}.jsonl").open("w") for s in ids_by_split}; report={}
    try:
      for split, wanted in ids_by_split.items():
        rows=[]; c=Counter(source_dialogues=len(wanted),missing_dialogues=len(wanted-set(by_id)))
        for sid in sorted(wanted):
          d=by_id.get(sid)
          if not d: continue
          turns=[u["text"].strip() for u in d.get("utterances",[]) if u.get("text","").strip()]
          if len(turns)<2 or any(not re.search(r"\w",t) for t in turns): c["filtered_dialogues"]+=1; continue
          rows.append((sid,turns)); c["accepted_dialogues"]+=1
        texts=[norm(t[0])+"".join("<ts> "+norm(x) for x in t[1:])+"<ts>" for _,t in rows]; enc=tok(texts,add_special_tokens=False,truncation=False)["input_ids"] if texts else []
        for (sid,turns),input_ids in zip(rows,enc):
          sp=[]; state=ids["speaker1"]
          for x in input_ids: sp.append(state); state=ids["speaker2"] if x==ids["ts"] and state==ids["speaker1"] else ids["speaker1"] if x==ids["ts"] else state
          c["turns"]+=len(turns); c["tokens"]+=len(input_ids); c["ts_tokens"]+=input_ids.count(ids["ts"]); stride=args.max_length-args.overlap_length
          for start in range(0,len(input_ids),stride):
            w=input_ids[start:start+args.max_length]
            if len(w)<args.keep_length: break
            handles[split].write(json.dumps({"dataset":"taskmaster1","source_split":split,"source_id":sid,"start_token":start,"input_ids":w,"speaker_ids":sp[start:start+args.max_length]},separators=(",",":"))+"\n"); c["windows"]+=1
        report[split]=dict(c)
    finally:
      [h.close() for h in handles.values()]
    manifest={"status":"ok","scope":"taskmaster1_one_person_dialogs","settings":{"max_length":args.max_length,"keep_length":args.keep_length,"overlap_length":args.overlap_length},"special_token_ids":ids,"splits":report}
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n"); print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
