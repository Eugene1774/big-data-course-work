from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI
from tqdm import tqdm

try:
    import fitz
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: pymupdf. Install with `pip install pymupdf`."
    ) from exc

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "knowledge_base" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "processed_json"
DEFAULT_CHUNK_SIZE = 4000
DEFAULT_MAX_CHARS = 3000
DEFAULT_MODE = "first"
DEFAULT_MAX_TOTAL_CHARS = 40000
DEFAULT_MAX_WORKERS = 5

DEFAULT_LLM_BASE_URL = (
    os.getenv("LLM_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or os.getenv("SILICONFLOW_BASE_URL")
    or "https://api.siliconflow.com/v1"
)
DEFAULT_LLM_MODEL = (
    os.getenv("LLM_MODEL")
    or os.getenv("OPENAI_MODEL")
    or os.getenv("SILICONFLOW_MODEL")
    or "Qwen/Qwen2.5-7B-Instruct"
)
DEFAULT_LLM_API_KEY = (
    os.getenv("LLM_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("SILICONFLOW_API_KEY")
)


PROMPT_TEMPLATE = """
You are a strict structured-information extraction assistant.

Task:
Extract key innovation information from the project application text and
return JSON only.

Hard requirements:
1) Output valid JSON only. No markdown, explanations, or comments.
2) The JSON must include exactly these fields:
   - project_name: string
   - technical_barriers: string[]
   - innovation_points: string[]
   - core_technology: string
   - application_scene: string
3) If a field is missing in the source, return empty string "" or empty list [].
   Do not fabricate.
4) technical_barriers and innovation_points must be concise and deduplicated.

Project application text:
{document_text}
""".strip()


LOGGER = logging.getLogger("batch_processor")
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def get_clean_pdf_list(folder_path: Path) -> List[Path]:
    """
    Filter valid PDF files:
    - keep only *.pdf (case-insensitive)
    - skip macOS metadata files starting with ._
    """
    if not folder_path.exists():
        return []
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {folder_path}")

    files: List[Path] = []
    for file_path in sorted(folder_path.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("._"):
            continue
        if file_path.suffix.lower() != ".pdf":
            continue
        files.append(file_path)
    return files


def extract_text_from_pdf(
    pdf_path: Path,
    max_chars: Optional[int] = None,
) -> str:
    """
    Extract text from PDF with per-file fault isolation.
    Stops early when max_chars is reached.
    """
    chunks: List[str] = []
    char_count = 0

    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            page_text = page.get_text("text") or ""
            if not page_text.strip():
                continue
            chunks.append(page_text)
            char_count += len(page_text)

            if max_chars is not None and char_count >= max_chars:
                break

    text = "\n".join(chunks).strip()
    if max_chars is not None:
        return text[:max_chars]
    return text


def split_text_by_chars(text: str, chunk_size: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not text:
        return []
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def parse_json_payload(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Empty response from model.")

    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    candidates: List[str] = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            pass

        cleaned = CONTROL_CHARS_RE.sub("", candidate)
        try:
            return json.loads(cleaned, strict=False)
        except json.JSONDecodeError:
            pass

    raise ValueError("Model response is not valid JSON even after fallback parsing.")


def build_openai_client(api_key: Optional[str], base_url: str) -> OpenAI:
    if not api_key:
        raise ValueError(
            "Missing API key. Set one of: LLM_API_KEY / OPENAI_API_KEY / SILICONFLOW_API_KEY."
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def call_llm_extract(
    client: OpenAI,
    model: str,
    text: str,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise data extraction assistant."},
            {"role": "user", "content": PROMPT_TEMPLATE.format(document_text=text)},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    content = response.choices[0].message.content or ""
    return parse_json_payload(content)


def normalize_extraction(payload: Dict[str, Any]) -> Dict[str, Any]:
    def _norm_str(value: Any) -> str:
        return str(value or "").strip()

    def _norm_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
        else:
            items = [str(value).strip()] if str(value).strip() else []
        return list(dict.fromkeys(items))

    return {
        "project_name": _norm_str(payload.get("project_name")),
        "technical_barriers": _norm_list(payload.get("technical_barriers")),
        "innovation_points": _norm_list(payload.get("innovation_points")),
        "core_technology": _norm_str(payload.get("core_technology")),
        "application_scene": _norm_str(payload.get("application_scene")),
    }


def merge_chunk_results(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    merged = {
        "project_name": "",
        "technical_barriers": [],
        "innovation_points": [],
        "core_technology": "",
        "application_scene": "",
    }

    for item in results:
        norm = normalize_extraction(item)
        for field in ("project_name", "core_technology", "application_scene"):
            if not merged[field] and norm[field]:
                merged[field] = norm[field]
        merged["technical_barriers"].extend(norm["technical_barriers"])
        merged["innovation_points"].extend(norm["innovation_points"])

    merged["technical_barriers"] = list(dict.fromkeys(merged["technical_barriers"]))
    merged["innovation_points"] = list(dict.fromkeys(merged["innovation_points"]))
    return merged


def process_single_pdf(
    pdf_path: Path,
    client: OpenAI,
    model: str,
    mode: str,
    max_chars: int,
    chunk_size: int,
    max_total_chars: Optional[int],
    temperature: float,
) -> Dict[str, Any]:
    if mode == "first":
        text = extract_text_from_pdf(pdf_path, max_chars=max_chars)
        if not text:
            raise ValueError("No extractable text found in PDF.")
        payload = call_llm_extract(client=client, model=model, text=text, temperature=temperature)
        return normalize_extraction(payload)

    text = extract_text_from_pdf(pdf_path, max_chars=max_total_chars)
    if not text:
        raise ValueError("No extractable text found in PDF.")

    chunks = split_text_by_chars(text, chunk_size=chunk_size)
    chunk_results: List[Dict[str, Any]] = []
    for chunk in chunks:
        payload = call_llm_extract(client=client, model=model, text=chunk, temperature=temperature)
        chunk_results.append(payload)

    return merge_chunk_results(chunk_results)


def process_single_file(
    pdf_path: Path,
    output_dir: Path,
    model: str,
    base_url: str,
    api_key: Optional[str],
    mode: str,
    max_chars: int,
    chunk_size: int,
    max_total_chars: Optional[int],
    temperature: float,
) -> Dict[str, Any]:
    output_path = output_dir / f"{pdf_path.stem}.json"
    if output_path.exists():
        return {"status": "skipped", "file": pdf_path.name}

    try:
        client = build_openai_client(api_key=api_key, base_url=base_url)
        extracted = process_single_pdf(
            pdf_path=pdf_path,
            client=client,
            model=model,
            mode=mode,
            max_chars=max_chars,
            chunk_size=chunk_size,
            max_total_chars=max_total_chars,
            temperature=temperature,
        )

        output_payload = {
            "source_file": pdf_path.name,
            "extraction": extracted,
        }
        output_path.write_text(
            json.dumps(output_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {"status": "success", "file": pdf_path.name}
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "file": pdf_path.name, "error": str(exc)}


def write_failure_report(output_dir: Path, failures: List[Dict[str, Any]]) -> Optional[Path]:
    if not failures:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"_failed_runs_{timestamp}.jsonl"
    with report_path.open("w", encoding="utf-8") as file_obj:
        for failure in failures:
            file_obj.write(json.dumps(failure, ensure_ascii=False) + "\n")
    return report_path


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    model: str,
    base_url: str,
    api_key: Optional[str],
    mode: str,
    max_chars: int,
    chunk_size: int,
    max_total_chars: Optional[int],
    temperature: float,
    max_workers: int,
) -> None:
    if max_workers <= 0:
        raise ValueError("max_workers must be > 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    _ = build_openai_client(api_key=api_key, base_url=base_url)

    pdf_files = get_clean_pdf_list(input_dir)
    to_process = [path for path in pdf_files if not (output_dir / f"{path.stem}.json").exists()]
    skipped_count = len(pdf_files) - len(to_process)

    LOGGER.info(
        "Detected %d valid PDF files under %s, pending=%d, skipped(existing)=%d",
        len(pdf_files),
        input_dir,
        len(to_process),
        skipped_count,
    )

    if not to_process:
        LOGGER.warning("No valid PDF files found. Nothing to process.")
        return

    success_count = 0
    fail_count = 0
    skipped_runtime_count = 0
    failures: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_single_file,
                pdf_path=pdf_path,
                output_dir=output_dir,
                model=model,
                base_url=base_url,
                api_key=api_key,
                mode=mode,
                max_chars=max_chars,
                chunk_size=chunk_size,
                max_total_chars=max_total_chars,
                temperature=temperature,
            )
            for pdf_path in to_process
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDFs", unit="file"):
            result = future.result()
            status = result.get("status")
            if status == "success":
                success_count += 1
            elif status == "failed":
                fail_count += 1
                failures.append(
                    {
                        "file": result.get("file"),
                        "error": result.get("error"),
                        "stage": "process_single_file",
                    }
                )
                LOGGER.error("Failed processing %s: %s", result.get("file"), result.get("error"))
            elif status == "skipped":
                skipped_runtime_count += 1

    failure_report_path = write_failure_report(output_dir=output_dir, failures=failures)
    LOGGER.info(
        "Pipeline finished. success=%d, failed=%d, skipped(existing)=%d, skipped(runtime)=%d",
        success_count,
        fail_count,
        skipped_count,
        skipped_runtime_count,
    )
    if failure_report_path is not None:
        LOGGER.info("Failure report written to %s", failure_report_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch process PDFs with LLM and export structured JSON."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Input PDF directory")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output JSON directory")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL, help="LLM model name")
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL, help="LLM API base URL")
    parser.add_argument("--api-key", default=DEFAULT_LLM_API_KEY, help="LLM API key")
    parser.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        choices=("first", "chunk"),
        help="first: analyze first N chars only; chunk: analyze all text by chunks.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Character limit in first mode.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size in chunk mode.",
    )
    parser.add_argument(
        "--max-total-chars",
        type=int,
        default=DEFAULT_MAX_TOTAL_CHARS,
        help="Total character cap before chunking. Use -1 for unlimited.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for LLM call.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum worker threads for concurrent processing.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def main() -> None:
    if load_dotenv is not None:
        load_dotenv()

    args = build_arg_parser().parse_args()
    configure_logging(verbose=args.verbose)

    max_total_chars: Optional[int]
    if args.max_total_chars is not None and int(args.max_total_chars) < 0:
        max_total_chars = None
    else:
        max_total_chars = int(args.max_total_chars)

    run_pipeline(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        mode=args.mode,
        max_chars=int(args.max_chars),
        chunk_size=int(args.chunk_size),
        max_total_chars=max_total_chars,
        temperature=float(args.temperature),
        max_workers=int(args.max_workers),
    )


if __name__ == "__main__":
    main()
