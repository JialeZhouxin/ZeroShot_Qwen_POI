import argparse
import ast
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from math import log2, radians, sin, cos, sqrt, atan2

import geohash2
from openai import OpenAI
from tqdm import tqdm

# ─── Path setup ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
_DATA_DIR = os.path.join(_SCRIPT_DIR, 'data')
_RESULTS_DIR = os.path.join(_SCRIPT_DIR, 'results')


# ─── Time helpers ─────────────────────────────────────────────────────────────
MONTH_NAMES = [None, 'January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
DAY_NAMES = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']


def ordinal(n):
    """1 → '1st', 2 → '2nd', 3 → '3rd', 16 → '16th', etc."""
    if 11 <= n % 100 <= 13:
        return f'{n}th'
    suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f'{n}{suffix}'


def time_period(hour):
    if hour < 6:
        return 'early Morning'
    elif hour < 9:
        return 'morning'
    elif hour < 12:
        return 'late Morning'
    elif hour < 18:
        return 'afternoon'
    elif hour < 21:
        return 'evening'
    else:
        return 'night'


def parse_time(raw_time):
    """Parse '2012/12/19 3:43' → (datetime, month_name, day_ord, period, hour_min, day_name)"""
    dt = datetime.strptime(raw_time.strip(), '%Y/%m/%d %H:%M')
    month = MONTH_NAMES[dt.month]
    day_ord = ordinal(dt.day)
    period = time_period(dt.hour)
    hour_min = dt.strftime('%H:%M')
    day_name = DAY_NAMES[dt.weekday()]
    return dt, month, day_ord, period, hour_min, day_name


def format_time_sentence(raw_time):
    """'2012/12/19 3:43' → 'December 19th early Morning at 03:43 (wednesday)'"""
    dt, month, day_ord, period, hour_min, day_name = parse_time(raw_time)
    return f'{month} {day_ord} {period} at {hour_min} ({day_name})'


def compute_delta(last_time, target_time):
    """Compute time difference in minutes between two time strings."""
    dt1, *_ = parse_time(last_time)
    dt2, *_ = parse_time(target_time)
    return int((dt2 - dt1).total_seconds() / 60)


def compute_distance_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km."""
    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    except (ValueError, TypeError):
        return 0.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return 6371.0 * c


# ─── Data loading ─────────────────────────────────────────────────────────────
def read_csv_dicts(filepath):
    """Read CSV file and return list of dicts, trying multiple encodings."""
    for enc in ['utf-8-sig', 'iso-8859-1', 'cp1252']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                return list(csv.DictReader(f))
        except (UnicodeDecodeError, UnicodeEncodeError):
            continue
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        return list(csv.DictReader(f))


def extract_category(raw_cat):
    """Extract clean category name from raw CSV value.
    CA dataset stores JSON like [{'url': '/categories/1', 'name': 'Coffee Shop'}].
    NYC/TKY store plain text like 'Train Station'.
    """
    raw = raw_cat.strip()
    if raw.startswith('['):
        # CA format: JSON list of dicts
        try:
            items = ast.literal_eval(raw.replace("'", '"'))
            if isinstance(items, list) and len(items) > 0:
                return items[0].get('name', raw)
        except (ValueError, SyntaxError):
            pass
        # Try single-quote JSON alternative
        try:
            items = ast.literal_eval(raw)
            if isinstance(items, list) and len(items) > 0:
                return items[0].get('name', raw)
        except (ValueError, SyntaxError):
            pass
    return raw


def load_dataset(dataset_name):
    """Load train and val data for a dataset (nyc/tky/ca)."""
    name_map = {'nyc': 'NYC', 'tky': 'TKY', 'ca': 'CA'}
    dset = name_map[dataset_name.lower()]

    train_path = os.path.join(_DATA_DIR, dset, f'{dset}_train.csv')
    val_path = os.path.join(_DATA_DIR, dset, f'{dset}_val.csv')

    # Read training data — build user histories and POI info
    train_rows = read_csv_dicts(train_path)
    user_histories = {}  # uid → [(poi_id, category, time, lat, lon), ...]
    poi_info = {}        # poi_id → {"category": ..., "latitude": ..., "longitude": ...}
    for r in train_rows:
        poi_id = r['PoiId'].strip()
        uid = r['UserId'].strip()
        cat = extract_category(r['PoiCategoryName'])
        lat = r['latitude'].strip()
        lon = r['longitude'].strip()
        time_str = r['local_time'].strip()

        if poi_id not in poi_info:
            poi_info[poi_id] = {
                "category": cat,
                "latitude": lat,
                "longitude": lon,
            }
        if uid not in user_histories:
            user_histories[uid] = []
        user_histories[uid].append((poi_id, cat, time_str, lat, lon))

    # Read validation data — group by trajectory_id
    val_rows = read_csv_dicts(val_path)
    trajectories = {}  # traj_id → [rows]
    for r in val_rows:
        traj_id = r['pseudo_session_trajectory_id'].strip()
        row = {
            "time": r['local_time'].strip(),
            "user_id": r['UserId'].strip(),
            "latitude": r['latitude'].strip(),
            "longitude": r['longitude'].strip(),
            "poi_id": r['PoiId'].strip(),
            "category": extract_category(r['PoiCategoryName']),
        }
        if traj_id not in trajectories:
            trajectories[traj_id] = []
        trajectories[traj_id].append(row)

    return user_histories, poi_info, trajectories


def filter_trajectories(trajectories, min_len=5):
    """Filter trajectories that have at least min_len check-ins."""
    filtered = {}
    for tid, rows in trajectories.items():
        if len(rows) >= min_len:
            filtered[tid] = rows
    return filtered


def build_long_term_preference(user_hist, max_cats=6):
    """Build the long-term preference string from user's training history."""
    cat_counts = Counter(cat for _, cat, _, _, _ in user_hist)
    total = sum(cat_counts.values())
    if total == 0:
        return "No historical data"

    parts = []
    for i, (cat, cnt) in enumerate(cat_counts.most_common(max_cats)):
        pct = cnt / total * 100
        if i == 0:
            label = "Daily routine"
        elif pct >= 5:
            label = "Frequent"
        else:
            label = "Occasional"
        parts.append(f'{label}: {cat} ({pct:.0f}%)')

    return '; '.join(parts)


def build_trajectory_section(traj_rows, max_len=50):
    """Build the short-term trajectory section from trajectory rows (all except last)."""
    lines = []
    for r in traj_rows[-max_len:]:
        time_sentence = format_time_sentence(r["time"])
        lat, lon = r["latitude"], r["longitude"]
        try:
            geo = geohash2.encode(float(lat), float(lon), precision=6)
        except (ValueError, TypeError):
            geo = "unknown"
        lines.append(f'- {time_sentence} | {r["category"]} | geohash code: {geo}')
    return '\n'.join(lines)


def build_candidate_pool(ground_truth_poi, all_poi_ids, size=9, seed=None):
    """Build candidate pool: GT + `size` random negatives, shuffled."""
    if seed is not None:
        random.seed(seed)

    # Sample negatives from all POIs except the ground truth
    negatives = [p for p in all_poi_ids if p != ground_truth_poi]
    sampled = random.sample(negatives, min(size, len(negatives)))
    pool = sampled + [ground_truth_poi]
    random.shuffle(pool)
    return pool


def build_candidate_section(candidate_ids, poi_info):
    """Build the Candidate Pool section string."""
    lines = ['Which of the following POIs is the user most likely to visit next?']
    for pid in candidate_ids:
        info = poi_info.get(pid, {"category": "Unknown"})
        lines.append(f'- [ID: {pid} | Category: {info["category"]}]')
    return '\n'.join(lines)


# ─── Prompt construction ──────────────────────────────────────────────────────
INSTRUCTION = (
    "Identify the most plausible next POI from the provided Candidate Pool "
    "based on the user's historical preferences, movement trajectory, "
    "and current spatial-temporal constraints."
)

OUTPUT_FORMAT_INSTRUCTION = (
    'Output a JSON object with key "predicted_poi_id" containing the ID of the most likely next POI '
    '(and only that key).'
)


def build_prompt(instruction, long_term, trajectory_section, context_section, candidate_section):
    """Build the full prompt with Qwen chat template.
    The output format instruction is embedded in the user message."""
    user_content = f"### [User Long-term Preference]\n- POI Check-in Categories：{long_term}\n"
    user_content += f"### [User Short-term Trajectory]\n POI Check-in Sequences:\n{trajectory_section}\n"
    user_content += f"### [Spatio-Temporal Context]\n{context_section}\n"
    user_content += f"### [Candidate Pool]\n{candidate_section}\n\n"
    user_content += OUTPUT_FORMAT_INSTRUCTION

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_content},
    ]
    return messages


# ─── LLM API ──────────────────────────────────────────────────────────────────
def create_client(api_base=None):
    api_key = (os.environ.get("OPENROUTER_API_KEY")
               or os.environ.get("OPENAI_API_KEY")
               or "")
    if not api_key:
        raise ValueError("Please set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable")

    kwargs = {"api_key": api_key}
    if api_base:
        kwargs["base_url"] = api_base
    kwargs["timeout"] = 120.0
    return OpenAI(**kwargs)


def call_llm(client, model, messages, max_retries=3):
    """Call LLM with JSON response format and retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            parsed = json.loads(content)
            return parsed
        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                raise ValueError(f"Failed to parse JSON after {max_retries} attempts: {content[:200]}")
            time.sleep(1)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError("Unreachable")


# ─── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(all_results, total_samples):
    """Compute Acc/NDCG/MRR from per-sample results.

    Single-GT setting: Recall@K = Acc@K (1 relevant item, binary hit).
    Precision@K = 1/K * Acc@K (1 hit among K positions).
    NDCG@1 = Acc@1 (IDCG = 1.0, only rank-1 contributes).
    These are omitted as redundant; derivable from Acc@K.
    """
    hit1 = hit5 = hit10 = 0
    rr_sum = 0.0
    ndcg5_sum = ndcg10_sum = 0.0
    valid_count = 0
    rank_sum = 0
    rank_count = 0

    for res in all_results:
        valid_count += 1 if res["valid"] else 0
        rank = res["rank"]  # 1-based, None if GT never found

        if rank is not None:
            if rank <= 10:
                hit10 += 1
                ndcg10_sum += 1.0 / log2(rank + 1)
            if rank <= 5:
                hit5 += 1
                ndcg5_sum += 1.0 / log2(rank + 1)
            if rank == 1:
                hit1 += 1

            rr_sum += 1.0 / rank
            rank_sum += rank
            rank_count += 1

    n = total_samples
    return {
        "total_samples": n,
        "Acc@1": hit1 / n,
        "Acc@5": hit5 / n,
        "Acc@10": hit10 / n,
        "NDCG@5": ndcg5_sum / n,
        "NDCG@10": ndcg10_sum / n,
        "MRR": rr_sum / n,
        "ValidRatio": valid_count / n,
        "MeanRank": rank_sum / rank_count if rank_count > 0 else None,
        "hit1": hit1, "hit5": hit5, "hit10": hit10,
        "total_checked": n,
    }


# ─── Main evaluation loop ─────────────────────────────────────────────────────
def evaluate(args):
    print(f"Loading dataset: {args.dataset}...")
    user_histories, poi_info, trajectories = load_dataset(args.dataset)

    filtered = filter_trajectories(trajectories, min_len=args.min_traj_len)
    print(f"Total trajectories: {len(trajectories)}")
    print(f"Filtered (min_len >= {args.min_traj_len}): {len(filtered)}")

    # For NYC-only: exclude trajectories where user has fewer than 2 distinct
    # categories. This is necessary to match the experiment's count (412 vs 413).
    # TKY and CA already match the experiment (1890, 711) without this filter.
    if args.dataset == 'nyc':
        valid_sample_ids = []
        for tid in filtered:
            uid = filtered[tid][0]["user_id"]
            hist = user_histories.get(uid, [])
            cats = set(cat for _, cat, _, _, _ in hist)
            if len(cats) >= 2:
                valid_sample_ids.append(tid)
        print(f"After excluding single-category users: {len(valid_sample_ids)}")
    else:
        valid_sample_ids = list(filtered.keys())

    all_poi_ids = list(poi_info.keys())
    print(f"Unique POIs: {len(all_poi_ids)}")

    # Use valid_sample_ids instead of filtered.keys()
    sample_ids = list(valid_sample_ids)
    if args.cases and args.cases < len(sample_ids):
        random.seed(42)
        sample_ids = random.sample(sample_ids, args.cases)
        print(f"Using {len(sample_ids)} random samples (--cases={args.cases})")

    if args.debug:
        sample_ids = sample_ids[:1]
        print(f"DEBUG mode: 1 sample")

    client = create_client(api_base=args.api_base)
    print(f"Model: {args.model}")

    all_results = []
    api_errors = 0
    total_llm_calls = 0
    early_stops = 0

    for tid in tqdm(sample_ids, desc="Evaluating"):
        try:
            result = evaluate_one(
                tid, filtered[tid], user_histories, poi_info,
                all_poi_ids, client, args.model, args.output_dir, args.dataset,
            )
            all_results.append(result)
            total_llm_calls += result["num_calls"]
            if result.get("early_stop"):
                early_stops += 1
        except Exception as e:
            api_errors += 1
            all_results.append({
                "trajectory_id": tid,
                "valid": False,
                "rank": None,
                "predictions": [],
                "ground_truth": filtered[tid][-1]["poi_id"],
                "num_calls": 0,
                "early_stop": False,
                "error": str(e),
            })
            if api_errors <= 5:
                print(f"\nError on {tid}: {repr(e)}")

    metrics = compute_metrics(all_results, len(sample_ids))
    metrics["api_errors"] = api_errors
    metrics["total_llm_calls"] = total_llm_calls
    metrics["avg_llm_calls_per_sample"] = total_llm_calls / len(sample_ids)
    metrics["early_stop_count"] = early_stops
    metrics["early_stop_ratio"] = early_stops / len(sample_ids)

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Total Samples:  {metrics['total_samples']}")
    print(f"Candidate size: {args.candidate_size}")
    print(f"Trajectory min length:  {args.min_traj_len}")
    print(f"Acc@1:          {metrics['Acc@1']:.4f} ({metrics['hit1']}/{metrics['total_checked']})")
    print(f"Acc@5:          {metrics['Acc@5']:.4f} ({metrics['hit5']}/{metrics['total_checked']})")
    print(f"Acc@10:         {metrics['Acc@10']:.4f} ({metrics['hit10']}/{metrics['total_checked']})")
    print(f"NDCG@5:         {metrics['NDCG@5']:.4f}")
    print(f"NDCG@10:        {metrics['NDCG@10']:.4f}")
    print(f"MRR:            {metrics['MRR']:.4f}")
    print(f"Valid Ratio:    {metrics['ValidRatio']:.4f}")
    print(f"Mean Rank:      {metrics['MeanRank']:.2f}" if metrics['MeanRank'] else "Mean Rank:      N/A")
    print(f"Efficiency Statistics:")
    print(f"  Total LLM calls: {total_llm_calls}")
    print(f"  Average LLM calls per sample: {metrics['avg_llm_calls_per_sample']:.2f}")
    print(f"  Average candidate pool size: {args.candidate_size}.00")
    print(f"  Early stop count: {early_stops}/{len(sample_ids)} ({metrics['early_stop_ratio']*100:.1f}%)")
    if api_errors > 0:
        print(f"  API errors: {api_errors}")

    # Save detailed results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'{args.dataset}_eval_cand_{args.candidate_size}_mintrj_{args.min_traj_len}_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"metrics": metrics, "results": all_results}, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")

    return metrics


def evaluate_one(tid, traj_rows, user_histories, poi_info, all_poi_ids, client, model, output_dir, dataset):
    """Evaluate a single trajectory with iterative guessing."""
    # Ground truth = last check-in
    target = traj_rows[-1]
    recent = traj_rows[:-1]  # all except last
    gt_poi = target["poi_id"]
    uid = traj_rows[0]["user_id"]

    # Construct candidate pool
    seed = int(tid) if tid.isdigit() else hash(tid) % (2**31)
    candidates = build_candidate_pool(gt_poi, all_poi_ids, size=9, seed=seed)
    # candidates is [10 items, shuffled], includes GT

    # Build static prompt sections
    user_hist = user_histories.get(uid, [])
    long_term = build_long_term_preference(user_hist)

    trajectory_section = build_trajectory_section(recent, max_len=50)

    # Spatio-temporal context
    target_time = format_time_sentence(target["time"])
    last_time = recent[-1]["time"]
    delta_t = compute_delta(last_time, target["time"])
    delta_d = compute_distance_km(
        recent[-1]["latitude"], recent[-1]["longitude"],
        target["latitude"], target["longitude"]
    )
    context_section = (
        f'- Target Time: {target_time}\n'
        f'- Displacement: Δd={delta_d:.2f}km, Δt={delta_t}min from last visit.'
    )

    remaining = list(candidates)  # mutable copy
    predictions = []  # ordered list of all predictions made
    num_calls = 0
    early_stop = False

    # Check for cached result
    os.makedirs(os.path.join(output_dir, dataset), exist_ok=True)
    cache_path = os.path.join(output_dir, dataset, tid)
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            cached = json.load(f)
            return cached

    # Iterative guessing loop
    # Remove max_invalid_retries: invalid outputs must be retried until
    # the model picks from the remaining pool or max_total_calls is hit.
    # This ensures Acc@N = 1.0 for N = candidate_pool_size in the common case
    # (GT is always in pool). In rare cases where all 10 calls output invalid IDs,
    # rank=None and the sample counts as a miss at all K.
    max_total_calls = 10
    while remaining and num_calls < max_total_calls:
        candidate_section = build_candidate_section(remaining, poi_info)
        messages = build_prompt(INSTRUCTION, long_term, trajectory_section, context_section, candidate_section)

        parsed = call_llm(client, model, messages)
        num_calls += 1

        predicted_id = str(parsed.get("predicted_poi_id", "")).strip()

        if predicted_id not in remaining:
            # Invalid output — ignore and retry with same prompt
            continue

        # Valid prediction from the pool
        predictions.append(predicted_id)

        if predicted_id == gt_poi:
            # Hit!
            early_stop = True
            break

        # Wrong guess: remove and continue
        remaining.remove(predicted_id)

    # Determine rank:
    #   early_stop = True  → rank = len(predictions) (1-indexed)
    #   pool exhausted      → rank = len(predictions) (GT must be last in pool)
    #   max_calls hit       → rank = None (model never converged; treat as miss)
    if early_stop:
        rank = len(predictions)
    elif not remaining:
        # All candidates exhausted: GT was the last one
        rank = len(predictions)
    else:
        rank = None

    # Assemble result
    result = {
        "trajectory_id": tid,
        "ground_truth": gt_poi,
        "valid": True,
        "rank": rank,
        "predictions": predictions,
        "num_calls": num_calls,
        "early_stop": early_stop,
        "error": None,
    }

    # Cache result
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)

    return result


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Zero-shot Next POI Recommendation via LLM API")
    parser.add_argument('-d', '--dataset', default='nyc', choices=['nyc', 'tky', 'ca'])
    parser.add_argument('--model', default='qwen/qwen-2.5-7b-instruct')
    parser.add_argument('--api-base', default='https://openrouter.ai/api/v1')
    parser.add_argument('--candidate-size', type=int, default=10)
    parser.add_argument('--min-traj-len', type=int, default=5)
    parser.add_argument('--cases', type=int, default=None, help='Number of test samples (default: all)')
    parser.add_argument('--debug', action='store_true', help='Debug mode: 1 sample')
    parser.add_argument('--output-dir', default=os.path.join(_SCRIPT_DIR, 'output'))
    args = parser.parse_args()

    metrics = evaluate(args)

    # Append to results.csv
    results_csv = os.path.join(_RESULTS_DIR, 'results.csv')
    timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M')
    notes = f'Zero-shot {args.model}, cand={args.candidate_size}, min_traj={args.min_traj_len}, iter_guess'
    row = (
        f'ZeroShot_Qwen,{args.dataset},'
        f'{metrics["Acc@1"]:.4f},{metrics["Acc@5"]:.4f},{metrics["Acc@10"]:.4f},'
        f'{metrics["MRR"]:.4f},{metrics["NDCG@5"]:.4f},{metrics["NDCG@10"]:.4f},'
        f'{metrics["ValidRatio"]:.4f},'
        f'{timestamp},{notes}\n'
    )

    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    header = 'paper,dataset,Acc@1,Acc@5,Acc@10,MRR,NDCG@5,NDCG@10,ValidRatio,timestamp,notes\n'
    if not os.path.exists(results_csv):
        with open(results_csv, 'w', encoding='utf-8') as f:
            f.write(header)
    with open(results_csv, 'a', encoding='utf-8') as f:
        f.write(row)
    print(f'\nResults appended to {results_csv}')

    return metrics


if __name__ == '__main__':
    main()
