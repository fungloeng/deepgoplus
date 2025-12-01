#!/bin/bash
set -e

# ==========================
# 默认参数
# ==========================
ONT="all"
EPOCHS=1
BATCH_SIZE=32
DEVICE="gpu:0"
DATA_DIR="galaxy"   # 默认数据目录
TERMS_FILE="$DATA_DIR/terms.pkl"

usage() {
  echo "Usage: $0 [-o mf|bp|cc|pf|all] [-e epochs] [-bs batch_size] [-d device] [--terms path] [-dd data_dir]"
  echo "Defaults: -o all -e 12 -bs 32 -d gpu:0 --terms $DATA_DIR/terms.pkl --data-dir galaxy"
}

# ==========================
# 解析参数
# ==========================
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--ont) ONT="$2"; shift 2;;
    -e|--epochs) EPOCHS="$2"; shift 2;;
    -bs|--batch-size) BATCH_SIZE="$2"; shift 2;;
    -d|--device) DEVICE="$2"; shift 2;;
    --terms) TERMS_FILE="$2"; shift 2;;
    -dd|--data-dir) DATA_DIR="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1"; usage; exit 1;;
  esac
done

mkdir -p "$DATA_DIR"
mkdir -p results
mkdir -p log

prepare_ontology() {
  local ont="$1"
  local ONT_UP=$(echo "$ont" | tr '[:lower:]' '[:upper:]')
  local GO_COL

  case "$ont" in
    mf) GO_COL="GO_MF_propagated";;
    bp) GO_COL="GO_BP_propagated";;
    cc) GO_COL="GO_CC_propagated";;
    pf) GO_COL="GO_PF_propagated";;
    *) echo "Unsupported ontology: $ont"; exit 1;;
  esac

  echo "[${ONT_UP}] Checking data and preparing pairs/PKL if missing..."

  # 生成 pairs 文件
  if [[ ! -f "$DATA_DIR/${ONT_UP}_train_pairs.tsv" ]] || [[ ! -f "$DATA_DIR/${ONT_UP}_test_pairs.tsv" ]] || [[ ! -f "$DATA_DIR/${ONT_UP}_validation_pairs.tsv" ]]; then
    python - <<PY
import csv, os
base = r"$DATA_DIR"
go_col = r"${GO_COL}"

def to_pairs(in_tsv, out_pairs):
    import re
    wrote = 0
    with open(in_tsv, newline='', encoding='utf-8') as fin, open(out_pairs, 'w', newline='', encoding='utf-8') as fout:
        reader = csv.DictReader(fin, delimiter='\t')
        for row in reader:
            acc = row.get('acc') or row.get('Entry') or row.get('Entry_clean')
            gos = (row.get(go_col) or '').strip()
            if not acc or not gos:
                continue
            # 支持分号和逗号作为分隔符，使用正则表达式分割
            # 按分号或逗号分割，支持连续的分隔符
            go_list = re.split(r'[;,]+', gos)
            for go in go_list:
                go = go.strip()
                if go:  # 确保不是空字符串，且不包含分隔符
                    # 再次清理，确保没有残留的分隔符
                    go = go.strip(';,').strip()
                    if go:
                        fout.write(f"{acc}\t{go}\n")
                        wrote += 1
    print(f"Wrote {wrote} pairs to {out_pairs}")

to_pairs(os.path.join(base, f"${ONT_UP}_train_data.tsv"), os.path.join(base, f"${ONT_UP}_train_pairs.tsv"))
to_pairs(os.path.join(base, f"${ONT_UP}_test_data.tsv"), os.path.join(base, f"${ONT_UP}_test_pairs.tsv"))
to_pairs(os.path.join(base, f"${ONT_UP}_validation_data.tsv"), os.path.join(base, f"${ONT_UP}_validation_pairs.tsv"))
PY
  fi

  # 生成 PKL 文件
  for split in train validation test; do
    if [[ ! -f "$DATA_DIR/${ONT_UP}_${split}_data.pkl" ]]; then
      python src/data/prepare_data.py \
        --go-file "$DATA_DIR/go.obo" \
        --data-file "$DATA_DIR/${ONT_UP}_${split}_pairs.tsv" \
        --sequences-file "$DATA_DIR/${ONT_UP}_${split}_sequences.fasta" \
        --out-file "$DATA_DIR/${ONT_UP}_${split}_data.pkl"
    fi
  done

  # 训练 DeepGOPlus
  echo "[${ONT_UP}] Training DeepGOPlus..."
  python deepgoplus/train.py \
    --go-file "$DATA_DIR/go.obo" \
    --train-data-file "$DATA_DIR/${ONT_UP}_train_data.pkl" \
    --valid-data-file "$DATA_DIR/${ONT_UP}_validation_data.pkl" \
    --test-data-file "$DATA_DIR/${ONT_UP}_test_data.pkl" \
    --terms-file "$TERMS_FILE" \
    --model-file "$DATA_DIR/model_${ont}.h5" \
    --out-file "$DATA_DIR/${ONT_UP}_predictions.pkl" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --split 0.9 \
    --device "$DEVICE"

  echo "[${ONT_UP}] Done. Model: $DATA_DIR/model_${ont}.h5, Predictions: $DATA_DIR/${ONT_UP}_predictions.pkl"
}

run_ont() {
  local ont="$1"
  prepare_ontology "$ont"
}

case "$ONT" in
  mf) run_ont mf;;
  bp) run_ont bp;;
  cc) run_ont cc;;
  pf) run_ont pf;;
  all)
    run_ont mf
    run_ont bp
    run_ont cc
    run_ont pf
    ;;
  *) echo "Invalid -o/--ont value: $ONT"; usage; exit 1;;
esac

echo "Training pipeline finished."
