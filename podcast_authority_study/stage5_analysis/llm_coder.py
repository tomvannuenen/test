"""
Stage 5: LLM-Assisted Coding
==============================
Applies the coding frame to segmented transcripts using Claude API.
Supports human-in-the-loop validation and intercoder reliability checks.

Usage:
    python -m stage5_analysis.llm_coder --api-key YOUR_KEY

Output:
    data/coded/{video_id}_coded.json
    data/coded_index.csv
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("WARNING: anthropic package not installed")


def load_coding_frame(frame_path: str = "config/coding_frame.yaml") -> dict:
    with open(frame_path) as f:
        return yaml.safe_load(f)


def build_system_prompt(coding_frame: dict) -> str:
    """Build the system prompt from the coding frame."""
    dimensions = coding_frame["coding_dimensions"]

    prompt = """You are a qualitative coding assistant for a media studies research project.
You are analyzing transcripts of right-wing and anti-establishment podcasts to understand
how hosts construct and legitimize their authority.

For each transcript segment you receive, apply the following coding frame.
Return ONLY valid JSON matching the schema below. Do not explain or narrate.

CODING DIMENSIONS:
"""
    for dim_name, dim_data in dimensions.items():
        prompt += f"\n## {dim_name}\n{dim_data['description']}\nValid codes:\n"
        for code, desc in dim_data["codes"].items():
            prompt += f"  - {code}: {desc}\n"

    prompt += """
RESPONSE SCHEMA (return exactly this JSON structure):
{
  "self_positioning": ["code1", "code2"],
  "enemy_construction": ["code1"],
  "epistemic_style": ["code1"],
  "power_model": ["code1", "code2"],
  "authorization_move": ["code1"],
  "authority_claim_summary": "Brief free-text summary of the authority claim being made in this segment",
  "confidence": 3,
  "notable_quotes": ["exact quote 1", "exact quote 2"],
  "notes": "Any analytical notes about ambiguity or interesting features"
}

Rules:
- Each dimension can have 0 or more codes. Use empty list [] if dimension is not present.
- confidence is 1-5 (1 = very uncertain, 5 = very clear coding).
- notable_quotes should be exact text from the segment (max 3).
- If the segment contains no relevant authority-related discourse (e.g., pure ad read,
  technical discussion with no political framing), return all empty lists and confidence 1.
- Be conservative: only code what is clearly present, not what is implied.
"""
    return prompt


def code_segment(client, system_prompt: str, segment_text: str,
                 channel_name: str, persona_type: str,
                 model: str = "claude-sonnet-4-20250514") -> dict | None:
    """Send a single segment to the LLM for coding."""
    user_message = f"""Channel: {channel_name} (persona type: {persona_type})

TRANSCRIPT SEGMENT:
{segment_text}

Apply the coding frame to this segment. Return only JSON."""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        # Parse response
        text = response.content[0].text.strip()
        # Handle potential markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        return json.loads(text)

    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"  API error: {e}")
        return None


def code_video(client, system_prompt: str, segments_path: str,
               model: str, sample_rate: float = 1.0) -> dict | None:
    """Code all windows in a segmented transcript file."""
    with open(segments_path) as f:
        doc = json.load(f)

    windows = doc.get("windows", [])
    if not windows:
        return None

    # Optional sampling for large corpora
    if sample_rate < 1.0:
        import random
        k = max(1, int(len(windows) * sample_rate))
        windows = random.sample(windows, k)

    coded_windows = []

    for window in windows:
        text = window.get("text", "")
        if not text or window.get("word_count", 0) < 50:
            continue

        # Truncate very long segments to avoid token limits
        if len(text) > 8000:
            text = text[:8000] + "... [truncated]"

        coding = code_segment(
            client, system_prompt, text,
            doc.get("channel_name", ""), doc.get("persona_type", ""),
            model=model,
        )

        if coding:
            coded_windows.append({
                "window_index": window["window_index"],
                "start_time": window["start_time"],
                "end_time": window["end_time"],
                "word_count": window["word_count"],
                "coding": coding,
            })

        time.sleep(0.5)  # Rate limiting

    result = {
        "video_id": doc["video_id"],
        "channel_name": doc.get("channel_name", ""),
        "persona_type": doc.get("persona_type", ""),
        "title": doc.get("title", ""),
        "publish_date": doc.get("publish_date", ""),
        "total_windows_coded": len(coded_windows),
        "coded_windows": coded_windows,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Stage 5: LLM-assisted coding")
    parser.add_argument("--api-key", required=True, help="Anthropic API key")
    parser.add_argument("--segments-dir", default="data/segments")
    parser.add_argument("--output-dir", default="data/coded")
    parser.add_argument("--index-output", default="data/coded_index.csv")
    parser.add_argument("--coding-frame", default="config/coding_frame.yaml")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--sample-rate", type=float, default=1.0,
                        help="Fraction of windows to code per video (default: 1.0 = all)")
    parser.add_argument("--channel", help="Code only this channel")
    parser.add_argument("--limit", type=int, help="Max videos to code")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if not HAS_ANTHROPIC:
        print("ERROR: anthropic package required. pip install anthropic")
        sys.exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Build coding system
    coding_frame = load_coding_frame(args.coding_frame)
    system_prompt = build_system_prompt(coding_frame)

    # Initialize client
    client = anthropic.Anthropic(api_key=args.api_key)

    # Find segment files
    segment_files = sorted(Path(args.segments_dir).glob("*_segments.json"))
    print(f"Found {len(segment_files)} segment files")

    if args.channel:
        filtered = []
        for sf in segment_files:
            with open(sf) as f:
                doc = json.load(f)
            if doc.get("channel_name") == args.channel:
                filtered.append(sf)
        segment_files = filtered
        print(f"  Filtered to {len(segment_files)} for channel: {args.channel}")

    if args.limit:
        segment_files = segment_files[: args.limit]

    index_rows = []

    for sf in tqdm(segment_files, desc="Coding"):
        video_id = sf.stem.replace("_segments", "")

        if args.skip_existing and os.path.exists(
            os.path.join(args.output_dir, f"{video_id}_coded.json")
        ):
            continue

        result = code_video(client, system_prompt, str(sf), args.model, args.sample_rate)

        if result is None:
            continue

        # Save coded file
        output_path = os.path.join(args.output_dir, f"{video_id}_coded.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        index_rows.append({
            "video_id": video_id,
            "channel_name": result["channel_name"],
            "persona_type": result["persona_type"],
            "windows_coded": result["total_windows_coded"],
        })

    # Write index
    with open(args.index_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video_id", "channel_name", "persona_type", "windows_coded",
        ])
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"\n{'=' * 60}")
    print(f"Coding complete: {len(index_rows)} videos coded")
    print(f"Index: {args.index_output}")


if __name__ == "__main__":
    main()
