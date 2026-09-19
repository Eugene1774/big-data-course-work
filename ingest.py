from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:
    from pydantic import BaseModel, Field

    ConfigDict = None

from schema.nodes import KnowledgeCard


LIST_SPLIT_PATTERN = re.compile(r"[，,、;/|；\n]+")
SENTENCE_SPLIT_PATTERN = re.compile(r"[。！？!\?\n]+")
BENCHMARK_CUES = ("基准", "标准", "建议", "通常", "一般", "推荐", "不低于", "不少于", "低于", "高于", "benchmark")

LABEL_KEYWORD_MAP = {
    "B2B": ["b2b", "企业客户", "企业采购", "采购负责人"],
    "工业设备": ["工业设备", "压缩机", "机床", "制造业", "招投标"],
    "渠道策略": ["渠道", "获客", "销售渠道", "distribution", "go to market"],
    "竞品分析": ["竞品", "对标", "benchmark", "competitor"],
    "单位经济": ["cac", "ltv", "unit economics", "回收周期"],
    "校园创业": ["高校", "校园", "大学生", "社团"],
    "教育科技": ["教育", "学习", "课程", "伴学"],
    "SaaS": ["saas", "订阅", "续费", "座席"],
}

SCENARIO_KEYWORD_MAP = {
    "工业设备销售": ["工业设备", "压缩机", "工厂采购", "招投标"],
    "高客单价B2B": ["高客单价", "企业采购", "采购决策", "企业客户"],
    "校园消费": ["校园", "高校", "大学生", "社团"],
    "教育科技": ["教育", "学习", "课程", "伴学"],
    "SaaS 增长": ["saas", "订阅", "续费", "座席"],
}

INDUSTRY_KEYWORD_MAP = {
    "AI": ["ai", "人工智能", "大模型", "机器学习", "算法", "智能体", "生成式"],
    "跨境电商": ["跨境电商", "出海电商", "亚马逊", "temu", "shein", "独立站", "海外仓"],
    "重工业": ["重工业", "工业设备", "机床", "压缩机", "工程机械", "制造业", "工厂采购"],
    "教育科技": ["教育科技", "edtech", "教育", "课程", "学习平台", "伴学"],
    "医疗健康": ["医疗", "健康", "医院", "诊疗", "药械", "医药"],
    "企业服务": ["企业服务", "saas", "企业客户", "b2b", "协同办公", "crm", "erp"],
}

CHANNEL_MAP = {
    "社交媒体": ["社交媒体", "社媒", "微博", "微信", "公众号", "小红书"],
    "短视频投流": ["短视频", "短视频投流", "抖音投流", "信息流投放", "直播投流"],
    "直销": ["直销", "销售拜访", "顾问销售", "顾问式销售"],
    "行业展会": ["行业展会", "展会", "博览会"],
    "招投标": ["招投标", "投标", "招标"],
    "线下顾问销售": ["线下顾问销售", "线下销售", "顾问式销售", "线下面谈"],
}

METRIC_ALIASES = {
    "CAC": ["cac", "获客成本", "客户获取成本"],
    "LTV": ["ltv", "客户生命周期价值", "客户终身价值"],
    "毛利率": ["毛利率"],
    "转化率": ["转化率"],
    "留存率": ["留存率"],
    "客单价": ["客单价"],
    "回收周期": ["回收周期", "cac 回收周期"],
}

PROJECT_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
DEFAULT_RAW_DIR = KNOWLEDGE_BASE_DIR / "raw"
DEFAULT_CARDS_DIR = KNOWLEDGE_BASE_DIR / "cards"
DEFAULT_LOG_FILE = PROJECT_ROOT / "ingest_log.txt"
IGNORED_FILE_NAMES = {".ds_store", "desktop.ini"}
SUPPORTED_DOCUMENT_SUFFIXES = {
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".txt",
    ".md",
    ".html",
    ".htm",
}
STRUCTURED_BINARY_SUFFIXES = {".pdf", ".doc", ".docx", ".ppt", ".pptx"}
GENERATED_CARD_INDEX_CACHE: Dict[str, int] = {}


class StrictBaseModel(BaseModel):
    """为本脚本统一提供严格字段校验配置。"""

    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    else:

        class Config:
            extra = "forbid"


class DecodedDocument(StrictBaseModel):
    """描述单份原始文献在解码与清洗后的中间结果。"""

    source_name: str = Field(...)
    source_path: str = Field(...)
    suffix: str = Field(...)
    parser: str = Field(...)
    encoding_used: str = Field(...)
    cleaned_text: str = Field(...)
    warnings: List[str] = Field(default_factory=list)
    source_locator: Optional[str] = Field(default=None)


class BenchmarkRecord(StrictBaseModel):
    """描述从文献中抽出的单条行业指标基准。"""

    metric_name: str = Field(...)
    original_text: str = Field(...)
    comparator: Optional[str] = Field(default=None)
    value: Optional[float] = Field(default=None)
    lower_bound: Optional[float] = Field(default=None)
    upper_bound: Optional[float] = Field(default=None)
    unit: Optional[str] = Field(default=None)


class ChannelFitRule(StrictBaseModel):
    """
    描述单个渠道在某行业/场景下的适配分值。

    设计原因：
    - 用户希望 `channel_fit_data` 直接输出为 H1 可调用的核心结构。
    - 因此这里不再把“推荐渠道”和“不推荐渠道”分别放成两个数组，
      而是把每个渠道拆成独立记录，并明确给出 `fit_score`。
    - 后续规则层只需要按渠道名读取分值，就能判断项目渠道是否匹配行业规律。
    """

    channel: str = Field(..., description="标准化后的渠道名称，例如 直销、社交媒体、行业展会")
    fit_score: float = Field(..., ge=0.1, le=1.0, description="渠道适配分值，越高表示越适合，越低表示越不适合")
    customer_segments: List[str] = Field(default_factory=list, description="如果原文提到了适用客户，则保存在这里")
    applicable_scenarios: List[str] = Field(default_factory=list, description="如果原文提到了适用场景，则保存在这里")
    reason: str = Field(default="", description="触发该分值判断的原文片段")


class IngestionResult(StrictBaseModel):
    """描述单份文献完成入库后的输出信息。"""

    source_path: str = Field(...)
    card_id: str = Field(...)
    output_path: Optional[str] = Field(default=None)
    warnings: List[str] = Field(default_factory=list)


class DiscoveryResult(StrictBaseModel):
    """
    描述一次目录扫描得到的文件发现结果。

    设计原因：
    - 用户不仅想知道“能处理哪些文档”，还想知道“有多少文件因为格式不支持被跳过”。
    - 因此这里把两类结果一起返回，主流程就能先打印总数，再开始真正导入。
    - 这样老师在面对超大文件夹时，可以先核对总数，确认没有漏扫。
    """

    document_files: List[Path] = Field(default_factory=list)
    skipped_non_document_files: List[Path] = Field(default_factory=list)


class UnreadableDocumentError(Exception):
    """
    表示原始文件无法被可靠识别或恢复。

    设计原因：
    - 批量导入时，目录中可能混入完全损坏的文件、二进制垃圾或临时缓存文件。
    - 这类文件不应该让整个导入流程中断。
    - 因此单独定义异常类型，便于主流程识别后写入 `ingest_log.txt` 并继续处理下一个文件。
    """


class SimpleHTMLStripper(HTMLParser):
    """把 HTML 文本转换成纯文本的轻量解析器。"""

    def __init__(self) -> None:
        """初始化内部文本缓冲区。"""

        super().__init__()
        self.chunks: List[str] = []

    def handle_data(self, data: str) -> None:
        """收集 HTML 标签之间的正文内容。"""

        if data.strip():
            self.chunks.append(data.strip())

    def get_text(self) -> str:
        """返回拼接后的纯文本结果。"""

        return "\n".join(self.chunks)


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """兼容不同 Pydantic 版本，把模型稳定序列化为字典。"""

    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def normalize_text(value: Any) -> str:
    """把任意输入规整成便于匹配的低噪声文本。"""

    if value is None:
        return ""
    text = str(value).replace("\ufeff", "").replace("\x00", " ").strip().lower()
    text = re.sub(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f]", " ", text)
    return " ".join(text.split())


def split_items(value: Any) -> List[str]:
    """把逗号、分号、换行混杂的列表文本拆成干净条目。"""

    if value is None:
        return []
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        items = [item.strip() for item in LIST_SPLIT_PATTERN.split(str(value)) if item.strip()]
    return list(dict.fromkeys(items))


def stable_identifier(source_name: str, text: str) -> str:
    """根据文件名和正文内容生成稳定且可重复的卡片 id。"""

    stem = re.sub(r"[^0-9A-Za-z]+", "_", Path(source_name).stem).strip("_").lower() or "doc"
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"knowledge_card_{stem}_{digest}"


def ensure_knowledge_base_directories() -> Tuple[Path, Path]:
    """
    自动创建知识库目录。

    设计目的：
    1. `knowledge_base/raw` 用来统一放老师提供的原始文献。
    2. `knowledge_base/cards` 用来统一放脚本生成的结构化知识卡片。
    3. 即使用户第一次运行项目，也不需要手动创建目录，降低使用门槛。
    """

    DEFAULT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_CARDS_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_RAW_DIR, DEFAULT_CARDS_DIR


def append_ingest_log(log_path: Path, level: str, source_path: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
    """
    把导入过程中的异常、跳过记录或重要告警写入日志文件。

    逻辑原理：
    - 批量导入时，最怕的是一份坏文件拖垮整个流程。
    - 因此脚本会把每次失败或跳过信息写成一行 JSON，方便后续追查。
    - 采用 JSON Lines 格式后，未来也能很方便接到数据看板或日志分析工具。
    """

    payload = {
        "level": level,
        "source_path": source_path,
        "message": message,
        "details": details or {},
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def shorten_display_name(file_name: str, max_length: int = 28) -> str:
    """
    缩短终端里显示的文件名，避免超长文件名把进度条挤乱。

    逻辑原理：
    - 老师上传的资料文件名可能很长，例如带课程名、年份、章节号。
    - 如果原样显示，终端里的进度条会非常难看，甚至换行。
    - 因此这里保留首尾关键信息，中间用省略号压缩。
    """

    normalized_name = file_name.strip()
    if len(normalized_name) <= max_length:
        return normalized_name

    head_length = max_length // 2 - 1
    tail_length = max_length - head_length - 1
    return f"{normalized_name[:head_length]}…{normalized_name[-tail_length:]}"


def build_terminal_progress_bar(current: int, total: int, width: int = 24) -> str:
    """
    构建一个不依赖第三方库的文本进度条。

    逻辑原理：
    - `current` 表示已经完成处理的文件数。
    - `total` 表示本次批量任务的总文件数。
    - 我们按比例填充 `#`，剩余部分用 `-`，形成简单直观的可视化效果。
    """

    safe_total = max(total, 1)
    bounded_current = max(0, min(current, safe_total))
    filled_width = int(width * bounded_current / safe_total)
    return f"[{'#' * filled_width}{'-' * (width - filled_width)}] {bounded_current}/{safe_total}"


def emit_ingest_progress(
    current: int,
    total: int,
    source_name: str,
    generated_count: int,
    skipped_count: int,
    failed_count: int,
) -> None:
    """
    把导入进度实时打印到终端。

    逻辑原理：
    - 进度信息统一写到 `stderr`，这样不会污染 `stdout` 上的 JSON 输出。
    - 如果终端支持原地刷新，就用 `\r` 更新同一行；否则退化成逐行打印。
    - 展示内容尽量贴近 0 代码用户的理解方式：当前在处理哪份文件、已生成多少张卡片。
    """

    progress_bar = build_terminal_progress_bar(current, total)
    display_name = shorten_display_name(source_name)
    message = f"{progress_bar} 正在处理《{display_name}》... 已生成 {generated_count} 张卡片"
    if skipped_count or failed_count:
        message += f" | 跳过 {skipped_count} | 失败 {failed_count}"

    if sys.stderr.isatty():
        sys.stderr.write("\r" + message.ljust(140))
    else:
        sys.stderr.write(message + "\n")
    sys.stderr.flush()


def emit_ingest_summary(total: int, generated_count: int, skipped_count: int, failed_count: int) -> None:
    """
    在批量导入结束后输出最终汇总信息。

    逻辑原理：
    - 进度条适合展示“进行中”的状态。
    - 任务结束后，还需要一条稳定的结果总结，帮助用户快速确认整体情况。
    - 因此这里统一给出处理总数、生成卡片数、跳过数和失败数。
    """

    progress_bar = build_terminal_progress_bar(total, total)
    message = (
        f"{progress_bar} 批量导入完成：共处理 {total} 份文件，"
        f"已生成 {generated_count} 张卡片"
        f" | 跳过 {skipped_count} | 失败 {failed_count}"
    )

    if sys.stderr.isatty():
        sys.stderr.write("\r" + message.ljust(140) + "\n")
    else:
        sys.stderr.write(message + "\n")
    sys.stderr.flush()


def emit_scan_discovery_summary(document_count: int, skipped_non_document_count: int) -> None:
    """
    在真正开始导入前，先把扫描总数打印给用户确认。

    逻辑原理：
    - 批量导入前，用户最关心的是“总共扫到了多少份文档”。
    - 如果目录里混有图片、压缩包、程序文件，也希望能明确知道它们被跳过了多少。
    - 因此这里专门输出两行简洁摘要，帮助用户快速核对文件夹规模。
    """

    sys.stderr.write(f"正在扫描文件夹... 发现共计 {document_count} 个待处理文档\n")
    if skipped_non_document_count > 0:
        sys.stderr.write(f"跳过了 {skipped_non_document_count} 个非文档文件（如图片或压缩包）\n")
    sys.stderr.flush()


def compress_page_numbers(page_numbers: Sequence[int]) -> str:
    """
    把页码列表压缩为便于人阅读的范围字符串。

    逻辑原理：
    - PDF 解析时往往会得到多个离散页码，例如 `[1, 2, 3, 6, 7]`。
    - 直接原样写入证据链会比较冗长，因此这里将连续页码压缩成 `1-3, 6-7`。
    - 这样后续老师或学生回查原文时，能更快定位证据出现的位置。
    """

    normalized_pages = sorted({page for page in page_numbers if page >= 1})
    if not normalized_pages:
        return ""

    ranges: List[str] = []
    start_page = normalized_pages[0]
    end_page = normalized_pages[0]

    for page_number in normalized_pages[1:]:
        if page_number == end_page + 1:
            end_page = page_number
            continue
        ranges.append(f"{start_page}-{end_page}" if start_page != end_page else str(start_page))
        start_page = page_number
        end_page = page_number

    ranges.append(f"{start_page}-{end_page}" if start_page != end_page else str(start_page))
    return ", ".join(ranges)


def looks_like_unreadable_binary(raw_bytes: bytes) -> bool:
    """
    粗略判断文件是否更像无法直接恢复的二进制垃圾，而不是普通文本。

    逻辑原理：
    - 如果空字节比例很高，或可打印字符比例极低，通常说明它不是可直接解码的正文材料。
    - 该函数只用于“提前止损”，避免对明显坏文件做无意义解析。
    - 对于 PDF、DOCX、PPTX 这类已知结构化二进制文件，会在上层逻辑中单独处理，不受这里影响。
    """

    if not raw_bytes:
        return True

    sample = raw_bytes[:4096]
    null_ratio = sample.count(0) / max(len(sample), 1)
    if null_ratio > 0.2:
        return True

    for encoding in ["utf-8-sig", "utf-8", "gb18030", "gbk", "utf-16"]:
        try:
            text = sample.decode(encoding, errors="strict")
        except Exception:
            continue

        normalized_text = normalize_text(text)
        meaningful_char_count = sum(1 for char in normalized_text if char.isalnum() or "\u4e00" <= char <= "\u9fff")
        if meaningful_char_count >= 8:
            return False

    printable_count = sum(1 for byte in sample if 32 <= byte <= 126 or byte in {9, 10, 13})
    printable_ratio = printable_count / max(len(sample), 1)
    return printable_ratio < 0.15


def generate_card_id(output_dir: Path) -> str:
    """
    按 `KC_001` 的格式生成下一个知识卡片编号。

    逻辑原理：
    - 教学场景里，老师和学生更容易记住顺序编号，而不是哈希值。
    - 因此这里会扫描 `knowledge_base/cards` 中已有的卡片文件，取当前最大编号后加一。
    - 这样既保证可读性，也能在重复运行时保持编号连续。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_key = str(output_dir.resolve())
    max_index = GENERATED_CARD_INDEX_CACHE.get(cache_key, 0)
    for path in output_dir.glob("*.json"):
        # 这里同时兼容两种历史命名：
        # 1. 旧格式：KC_001.json
        # 2. 新格式：KC_重工业_渠道策略_001.json
        # 这样无论用户此前生成过哪种文件，序号都能连续累加。
        match = re.search(r"(?:^KC_|_)(\d+)$", path.stem, re.IGNORECASE)
        if match:
            max_index = max(max_index, int(match.group(1)))
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            candidate = str(payload.get("card_id") or payload.get("id") or "")
            match = re.search(r"KC_(\d+)", candidate, re.IGNORECASE)
            if match:
                max_index = max(max_index, int(match.group(1)))
        except Exception:
            continue
    max_index += 1
    GENERATED_CARD_INDEX_CACHE[cache_key] = max_index
    return f"KC_{max_index:03d}"


def sanitize_filename_component(value: Optional[str], fallback: str) -> str:
    """
    把行业名、类型名这类文本转换为安全的文件名片段。

    逻辑原理：
    - Windows 文件名不允许出现 `\\ / : * ? " < > |` 等字符。
    - 但我们的行业与类型字段经常来自原始文献，可能包含这些符号。
    - 因此这里统一做清洗，既保留中文可读性，又避免写文件时报错。
    """

    normalized_value = (value or "").strip()
    if not normalized_value:
        normalized_value = fallback

    normalized_value = re.sub(r'[\\/:*?"<>|]+', "_", normalized_value)
    normalized_value = re.sub(r"\s+", "_", normalized_value)
    normalized_value = normalized_value.strip("._ ")
    return normalized_value or fallback


def build_output_filename(card: KnowledgeCard) -> str:
    """
    按 `KC_行业_类型_序号.json` 的格式生成知识卡片文件名。

    逻辑原理：
    - `card_id` 继续承担系统内部的稳定编号角色，例如 `KC_001`。
    - 文件名则更偏向人类可读，方便 0 代码用户在文件夹里直接按行业和类型浏览。
    - 因此这里把“行业 + 类型 + 序号”拼到文件名里，同时保留序号唯一性。
    """

    sequence_match = re.search(r"(\d+)$", card.card_id or card.id)
    sequence = sequence_match.group(1) if sequence_match else "000"
    industry_part = sanitize_filename_component(card.industry, "未知行业")
    type_part = sanitize_filename_component(card.type, "行业基准")
    return f"KC_{industry_part}_{type_part}_{sequence}.json"


def classify_card_type(text: str, labels: Sequence[str], scenarios: Sequence[str], industry_benchmarks: Dict[str, Any]) -> str:
    """
    把文献自动归类为“政策标准”“行业基准”“失败案例”或“渠道策略”。

    逻辑原理：
    - 这一步相当于对原始文献做一次轻量语义建模。
    - 我们综合标题、正文、标签、场景和抽出的结构化内容，推断它最像哪一类知识卡。
    - 分类结果会直接影响后续教学智能体检索和提示时采用的解释口径。
    """

    normalized_text = normalize_text(" ".join([text, " ".join(labels), " ".join(scenarios)]))
    if any(keyword in normalized_text for keyword in ["政策", "条例", "办法", "规范", "指南", "国标", "标准"]):
        return "政策标准"
    if any(keyword in normalized_text for keyword in ["失败", "复盘", "教训", "踩坑", "倒闭", "误判", "谬误"]):
        return "失败案例"
    if "channel_fit_data" in industry_benchmarks or any(keyword in normalized_text for keyword in ["渠道", "获客", "投流", "展会", "直销"]):
        return "渠道策略"
    return "行业基准"


def detect_encoding_with_chardet(raw_bytes: bytes) -> Tuple[Optional[str], List[str]]:
    """
    使用 `chardet` 识别文本编码。

    逻辑原理：
    - 对于老师提供的原始文献，最常见的问题就是编码来源不统一。
    - `chardet` 会根据字节分布猜测最可能的编码和置信度。
    - 如果库不存在或置信度较低，函数不会中断流程，而是返回告警并交给后续回退策略处理。
    """

    warnings: List[str] = []
    try:
        import chardet  # type: ignore
    except ImportError:
        warnings.append("未安装 chardet，已回退到内置编码尝试策略。")
        return None, warnings

    result = chardet.detect(raw_bytes)
    encoding = result.get("encoding")
    confidence = float(result.get("confidence") or 0.0)

    if not encoding:
        warnings.append("chardet 未能识别出可靠编码，已回退到内置编码尝试策略。")
        return None, warnings

    if confidence < 0.6:
        warnings.append(f"chardet 编码识别置信度较低（{confidence:.2f}），将继续尝试多个常见编码。")

    return encoding, warnings


def decode_bytes_safely(raw_bytes: bytes) -> Tuple[str, str, List[str]]:
    """
    尽量从混乱编码中恢复文本，并返回采用的编码和告警。

    逻辑原理：
    1. 先调用 `chardet` 作为首选编码猜测器。
    2. 如果严格解码失败，就按常见中文和西文编码依次回退。
    3. 若仍失败，则使用 `replace` 模式做最后抢救，并选择乱码最少的结果。
    这样可以最大程度避免中文文献出现乱码。
    """

    if not raw_bytes:
        return "", "utf-8", []

    warnings: List[str] = []
    detected_encoding, detection_warnings = detect_encoding_with_chardet(raw_bytes)
    warnings.extend(detection_warnings)

    encodings: List[str] = []
    if detected_encoding:
        encodings.append(detected_encoding)
    encodings.extend(["utf-8-sig", "utf-8", "gb18030", "gbk", "big5", "utf-16", "utf-16-le", "utf-16-be", "latin-1"])

    deduplicated_encodings: List[str] = []
    for encoding in encodings:
        normalized_encoding = encoding.lower()
        if normalized_encoding not in deduplicated_encodings:
            deduplicated_encodings.append(normalized_encoding)

    for encoding in deduplicated_encodings:
        try:
            text = raw_bytes.decode(encoding, errors="strict")
            return text, encoding, warnings
        except UnicodeDecodeError:
            warnings.append(f"使用编码 {encoding} 严格解码失败，已尝试下一种编码。")
        except LookupError:
            warnings.append(f"编码 {encoding} 在当前环境不可用，已尝试下一种编码。")

    best_text, best_encoding, best_score = "", "latin-1", 10**9
    for encoding in deduplicated_encodings:
        try:
            text = raw_bytes.decode(encoding, errors="replace")
        except LookupError:
            continue
        score = (
            text.count("�") * 20
            + text.count("锟斤拷") * 10
            + text.count("Ã") * 5
            + text.count("Â") * 5
            + text.count("\x00") * 5
        )
        if score < best_score:
            best_text, best_encoding, best_score = text, encoding, score
        if score == 0:
            break

    if best_text.count("�") or best_text.count("锟斤拷"):
        warnings.append("原始文献存在乱码或损坏字节，已尽力修复。")

    return best_text, best_encoding, warnings


def clean_text(text: str) -> str:
    """清理多余空白、空字节和常见乱码占位符。"""

    cleaned = text.replace("�", " ").replace("锟斤拷", " ")
    cleaned = re.sub(r"\r\n?", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return "\n".join(line.strip() for line in cleaned.splitlines() if line.strip())


def extract_text_from_docx(raw_bytes: bytes) -> Tuple[str, str, List[str]]:
    """从 docx 压缩包中提取段落文本，失败时回退为空文本。"""

    warnings: List[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
            xml_bytes = archive.read("word/document.xml")
        root = ElementTree.fromstring(xml_bytes)
        texts = [node.text for node in root.iter() if node.text]
        return "\n".join(texts), "docx", warnings
    except Exception:
        warnings.append("DOCX 解析失败，已回退为普通文本解码。")
        decoded, encoding, decode_warnings = decode_bytes_safely(raw_bytes)
        return decoded, f"docx-fallback:{encoding}", warnings + decode_warnings


def extract_text_from_pptx(raw_bytes: bytes) -> Tuple[str, str, List[str]]:
    """
    从 PPTX 压缩包中提取幻灯片文本。

    逻辑原理：
    - PPTX 和 DOCX 一样，本质上是 ZIP + XML。
    - 对教学项目资料来说，很多关键信息会直接写在答辩 PPT 里。
    - 因此这里直接读取 `ppt/slides/*.xml` 中的文本节点，作为非结构化抽取输入。
    """

    warnings: List[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
            slide_names = sorted(name for name in archive.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml"))
            texts: List[str] = []
            for slide_name in slide_names:
                root = ElementTree.fromstring(archive.read(slide_name))
                texts.extend(node.text for node in root.iter() if node.text)
        return "\n".join(texts), "pptx", warnings
    except Exception:
        warnings.append("PPTX 解析失败，已回退为普通文本解码。")
        decoded, encoding, decode_warnings = decode_bytes_safely(raw_bytes)
        return decoded, f"pptx-fallback:{encoding}", warnings + decode_warnings


def extract_text_from_legacy_office(raw_bytes: bytes, file_type: str) -> Tuple[str, str, List[str]]:
    """
    对旧版 `.doc` / `.ppt` 做兼容性文本抢救。

    逻辑原理：
    - 旧版 Office 文件通常是二进制容器，不像 `.docx` / `.pptx` 那样容易直接解析。
    - 在不额外安装重型依赖的前提下，这里采用“尽量抢救可读文本”的策略。
    - 如果文件内部包含可解码文本片段，脚本会尽量提取出来；即使不完美，也比直接漏掉更友好。
    """

    warnings: List[str] = [f"{file_type.upper()} 为旧版二进制格式，已启用兼容抢救模式。"]

    # 先尝试按常见编码直接解码，适合那些其实是导出文本或结构较简单的文件。
    decoded_text, encoding_used, decode_warnings = decode_bytes_safely(raw_bytes)

    # 再补一层“字符串打捞”，把连续可打印 ASCII 片段提取出来，
    # 这样即便整体解码不理想，也能保住一部分标题、表头、链接等信息。
    ascii_chunks = [
        match.decode("latin-1", errors="ignore")
        for match in re.findall(rb"[\x20-\x7E]{6,}", raw_bytes)
    ]
    rescued_ascii_text = "\n".join(chunk.strip() for chunk in ascii_chunks if chunk.strip())

    merged_text = "\n".join(part.strip() for part in [decoded_text, rescued_ascii_text] if part and part.strip())
    return merged_text or decoded_text, f"{file_type}-fallback:{encoding_used}", warnings + decode_warnings


def extract_text_from_html(text: str) -> Tuple[str, str]:
    """从 HTML 文本中移除标签，只保留主体内容。"""

    stripper = SimpleHTMLStripper()
    stripper.feed(text)
    return stripper.get_text(), "html"


def extract_text_from_pdf(raw_bytes: bytes) -> Tuple[str, str, List[str], Optional[str]]:
    """
    优先用 `pdfplumber` 解析 PDF，缺库或失败时回退到文本抢救策略。

    逻辑原理：
    - PDF 的文本层结构通常比普通文本更复杂，直接按字节解码容易得到噪声。
    - `pdfplumber` 对常见教学资料、报告类 PDF 的正文抽取更稳定。
    - 若 PDF 本身没有可提取文本层，函数会回退到字节抢救模式，尽可能保留残余可读内容。
    """

    warnings: List[str] = []
    try:
        import pdfplumber  # type: ignore

        pages_text: List[str] = []
        non_empty_pages: List[int] = []
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            for page in pdf.pages:
                try:
                    page_text = page.extract_text() or ""
                    if not page_text.strip():
                        page_text = page.extract_text(keep_blank_chars=True) or ""
                    if not page_text.strip():
                        page_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                    
                    pages_text.append(page_text)
                    if page_text.strip():
                        non_empty_pages.append(page.page_number)
                except Exception:
                    pages_text.append("")
        extracted_text = "\n".join(pages_text)
        if extracted_text.strip():
            page_hint = compress_page_numbers(non_empty_pages)
            source_locator = f"页码: {page_hint}" if page_hint else None
            return extracted_text, "pdfplumber", warnings, source_locator
        warnings.append("pdfplumber 未提取到正文文本，正在尝试其他方法...")
        
        try:
            import PyPDF2  # type: ignore
            
            pages_text = []
            non_empty_pages = []
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(raw_bytes))
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text() or ""
                    pages_text.append(page_text)
                    if page_text.strip():
                        non_empty_pages.append(page_num + 1)
                except Exception:
                    pages_text.append("")
            
            extracted_text = "\n".join(pages_text)
            if extracted_text.strip():
                page_hint = compress_page_numbers(non_empty_pages)
                source_locator = f"页码: {page_hint}" if page_hint else None
                warnings.append("使用 PyPDF2 作为备选解析器。")
                return extracted_text, "pypdf2", warnings, source_locator
            warnings.append("PyPDF2 也未提取到正文文本，已使用回退策略。")
        except ImportError:
            warnings.append("未安装 PyPDF2，已使用回退策略。")
        except Exception as e:
            warnings.append(f"PyPDF2 解析失败: {str(e)[:50]}，已使用回退策略。")
    except ImportError:
        warnings.append("未安装 pdfplumber，已使用回退策略。")
    except Exception as e:
        warnings.append(f"pdfplumber 解析失败: {str(e)[:50]}，已使用回退策略。")

    decoded, encoding, decode_warnings = decode_bytes_safely(raw_bytes)
    rescued = "\n".join(match for match in re.findall(r"\(([^\(\)]{4,})\)", decoded) if not match.startswith("http"))
    return rescued or decoded, f"pdf-fallback:{encoding}", warnings + decode_warnings, None


def load_document(source_path: Path) -> DecodedDocument:
    """根据扩展名选择合适解析器，得到清洗后的文献正文。"""

    raw_bytes = source_path.read_bytes()
    suffix = source_path.suffix.lower()
    warnings: List[str] = []
    source_locator: Optional[str] = None
    if suffix not in STRUCTURED_BINARY_SUFFIXES and looks_like_unreadable_binary(raw_bytes):
        raise UnreadableDocumentError("文件内容更像无法直接恢复的二进制数据，已跳过。")

    if suffix == ".docx":
        text, parser, parser_warnings = extract_text_from_docx(raw_bytes)
        encoding_used = parser
    elif suffix == ".doc":
        text, parser, parser_warnings = extract_text_from_legacy_office(raw_bytes, "doc")
        encoding_used = parser
    elif suffix == ".pptx":
        text, parser, parser_warnings = extract_text_from_pptx(raw_bytes)
        encoding_used = parser
    elif suffix == ".ppt":
        text, parser, parser_warnings = extract_text_from_legacy_office(raw_bytes, "ppt")
        encoding_used = parser
    elif suffix in {".html", ".htm"}:
        decoded, encoding_used, parser_warnings = decode_bytes_safely(raw_bytes)
        text, parser = extract_text_from_html(decoded)
    elif suffix == ".pdf":
        text, parser, parser_warnings, source_locator = extract_text_from_pdf(raw_bytes)
        encoding_used = parser
    else:
        text, encoding_used, parser_warnings = decode_bytes_safely(raw_bytes)
        parser = "text"
    warnings.extend(parser_warnings)
    return DecodedDocument(
        source_name=source_path.name,
        source_path=str(source_path),
        suffix=suffix or ".txt",
        parser=parser,
        encoding_used=encoding_used,
        cleaned_text=clean_text(text),
        warnings=warnings,
        source_locator=source_locator,
    )


def load_document_bytes(source_name: str, raw_bytes: bytes, source_path: Optional[str] = None) -> DecodedDocument:
    """按文件名和字节流复用现有解析流程，返回清洗后的纯文本结果。"""

    suffix = Path(source_name).suffix.lower()
    warnings: List[str] = []
    source_locator: Optional[str] = None
    if suffix not in STRUCTURED_BINARY_SUFFIXES and looks_like_unreadable_binary(raw_bytes):
        raise UnreadableDocumentError("鏂囦欢鍐呭鏇村儚鏃犳硶鐩存帴鎭㈠鐨勪簩杩涘埗鏁版嵁锛屽凡璺宠繃銆?")

    if suffix == ".docx":
        text, parser, parser_warnings = extract_text_from_docx(raw_bytes)
        encoding_used = parser
    elif suffix == ".doc":
        text, parser, parser_warnings = extract_text_from_legacy_office(raw_bytes, "doc")
        encoding_used = parser
    elif suffix == ".pptx":
        text, parser, parser_warnings = extract_text_from_pptx(raw_bytes)
        encoding_used = parser
    elif suffix == ".ppt":
        text, parser, parser_warnings = extract_text_from_legacy_office(raw_bytes, "ppt")
        encoding_used = parser
    elif suffix in {".html", ".htm"}:
        decoded, encoding_used, parser_warnings = decode_bytes_safely(raw_bytes)
        text, parser = extract_text_from_html(decoded)
    elif suffix == ".pdf":
        text, parser, parser_warnings, source_locator = extract_text_from_pdf(raw_bytes)
        encoding_used = parser
    else:
        text, encoding_used, parser_warnings = decode_bytes_safely(raw_bytes)
        parser = "text"

    warnings.extend(parser_warnings)
    return DecodedDocument(
        source_name=Path(source_name).name,
        source_path=source_path or source_name,
        suffix=suffix or ".txt",
        parser=parser,
        encoding_used=encoding_used,
        cleaned_text=clean_text(text),
        warnings=warnings,
        source_locator=source_locator,
    )


def extract_explicit_items(text: str, field_names: Sequence[str]) -> List[str]:
    """从“标签：”“适用场景：”这类显式字段中抓取列表值。"""

    pattern = re.compile(rf"^(?:{'|'.join(re.escape(name) for name in field_names)})\s*[:：]\s*(.+)$", re.IGNORECASE)
    results: List[str] = []
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            results.extend(split_items(match.group(1)))
    return list(dict.fromkeys(results))


def extract_explicit_value(text: str, field_names: Sequence[str]) -> Optional[str]:
    """
    从文本中提取单值型显式字段。

    逻辑原理：
    - 某些字段不适合按列表切分，例如“出处”“来源”“页码”。
    - 如果仍沿用列表提取，会把一整条参考来源拆碎，导致证据链失真。
    - 因此这里保留原始字段右侧文本，尽量原样传递给 `evidence_source`。
    """

    pattern = re.compile(rf"^(?:{'|'.join(re.escape(name) for name in field_names)})\s*[:：]\s*(.+)$", re.IGNORECASE)
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            value = match.group(1).strip()
            if value:
                return value
    return None


def infer_labels(text: str) -> List[str]:
    """综合显式关键词行和全文关键词，推断知识卡片标签。"""

    labels = extract_explicit_items(text, ["标签", "关键词", "关键字", "主题词", "labels", "keywords"])
    normalized_text = normalize_text(text)
    for label, keywords in LABEL_KEYWORD_MAP.items():
        if any(keyword in normalized_text for keyword in keywords):
            labels.append(label)
    return list(dict.fromkeys(labels))


def infer_scenarios(text: str, labels: Sequence[str]) -> List[str]:
    """综合显式场景字段和关键词，推断文献的适用场景。"""

    scenarios = extract_explicit_items(text, ["适用场景", "应用场景", "适用于", "适用对象", "scenarios"])
    normalized_text = normalize_text(text + " " + " ".join(labels))
    for scenario, keywords in SCENARIO_KEYWORD_MAP.items():
        if any(keyword in normalized_text for keyword in keywords):
            scenarios.append(scenario)
    return list(dict.fromkeys(scenarios))


def infer_industry(text: str, labels: Sequence[str], scenarios: Sequence[str]) -> Optional[str]:
    """
    识别文献所属行业。

    逻辑原理：
    - 优先读取老师已显式标注的“行业/所属行业”字段，避免覆盖人工判断。
    - 若原文未显式给出，再结合正文、标签、适用场景做关键词打分。
    - 返回得分最高的行业名称；若完全没有足够信号，则返回空值，避免误判。
    """

    explicit_industry = extract_explicit_value(text, ["行业", "所属行业", "industry"])
    if explicit_industry:
        return explicit_industry

    normalized_text = normalize_text(text + " " + " ".join(labels) + " " + " ".join(scenarios))
    best_match: Optional[str] = None
    best_score = 0

    for industry_name, keywords in INDUSTRY_KEYWORD_MAP.items():
        score = sum(1 for keyword in keywords if keyword in normalized_text)
        if score > best_score:
            best_match = industry_name
            best_score = score

    return best_match if best_score > 0 else None


def extract_summary(text: str, fallback_title: str) -> str:
    """优先抽取摘要段，否则退回到文首自然段作为卡片说明。"""

    abstract_match = re.search(r"(?:摘要|abstract)\s*[:：]\s*([^\n]{20,400})", text, re.IGNORECASE)
    if abstract_match:
        return " ".join(abstract_match.group(1).split())[:280]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    summary = " ".join(lines[:3])[:280]
    return summary or fallback_title


def build_evidence_source(document: DecodedDocument) -> str:
    """
    构建知识卡片的证据链来源字段。

    逻辑原理：
    - `evidence_source` 的目标不是摘要，而是帮助人回到原文。
    - 因此优先保留老师手工写入的“出处/来源/页码”字段。
    - 若原文没有显式出处，则至少保留文件路径；若 PDF 已识别到页码，也一并写入。
    """

    explicit_source = extract_explicit_value(
        document.cleaned_text,
        ["出处", "来源", "原文出处", "文献出处", "证据来源", "source", "参考来源", "参考文献"],
    )
    explicit_page = extract_explicit_value(
        document.cleaned_text,
        ["页码", "引用页码", "page", "pages", "page_range"],
    )

    parts: List[str] = []
    if explicit_source:
        parts.append(f"原文出处: {explicit_source}")
    else:
        parts.append(f"文件来源: {document.source_path}")

    if explicit_page:
        parts.append(f"页码: {explicit_page}")
    elif document.source_locator:
        parts.append(document.source_locator)

    return " | ".join(parts)


def detect_channels(text: str) -> List[str]:
    """在一句话中识别渠道名称，并统一映射到规范渠道词。"""

    normalized_text = normalize_text(text)
    channels: List[str] = []
    for canonical_name, aliases in CHANNEL_MAP.items():
        if any(alias in normalized_text for alias in aliases):
            channels.append(canonical_name)
    return list(dict.fromkeys(channels))


def merge_channel_fit_record(existing_record: Optional[ChannelFitRule], new_record: ChannelFitRule) -> ChannelFitRule:
    """
    合并同一渠道的多次命中记录，避免生成重复渠道项。

    逻辑原理：
    - 一篇文献里同一渠道可能被多次提到，例如既说“直销更适合”，又说“行业展会和直销优先”。
    - 为了让输出更紧凑，我们按渠道名聚合，并保留更强的适配分值。
    - `reason` 会尽量拼接前几条原文依据，方便后续人工复核。
    """

    if existing_record is None:
        return new_record

    merged_reasons = [reason for reason in [existing_record.reason, new_record.reason] if reason]
    merged_reason = "；".join(list(dict.fromkeys(merged_reasons))[:3])
    existing_distance = abs(existing_record.fit_score - 0.5)
    new_distance = abs(new_record.fit_score - 0.5)

    # 如果同一渠道既被正向提到又被反向提到，优先保留“更强烈”的判断。
    # 当强度相同时，默认选择更保守的较低分值，避免把可疑渠道误判成高适配。
    selected_fit_score = existing_record.fit_score
    if new_distance > existing_distance:
        selected_fit_score = new_record.fit_score
    elif new_distance == existing_distance:
        selected_fit_score = min(existing_record.fit_score, new_record.fit_score)

    return ChannelFitRule(
        channel=existing_record.channel,
        fit_score=selected_fit_score,
        customer_segments=list(dict.fromkeys(existing_record.customer_segments + new_record.customer_segments)),
        applicable_scenarios=list(dict.fromkeys(existing_record.applicable_scenarios + new_record.applicable_scenarios)),
        reason=merged_reason,
    )


def extract_channel_fit_rules(text: str, scenarios: Sequence[str]) -> List[ChannelFitRule]:
    """
    从文献中抽取渠道适配分值，供 H1 直接调用。

    输出格式示例：
    - `{"channel": "直销", "fit_score": 0.9}` 表示行业较适合该渠道。
    - `{"channel": "社交媒体", "fit_score": 0.1}` 表示行业明显不适合该渠道。

    逻辑原理：
    - 我们先识别“适合/推荐/优先”这类正向提示词，以及“不适合/避免/慎用”这类反向提示词。
    - 然后把命中的渠道拆成单独记录，并赋予统一的分值。
    - 分值并不是精确统计学概率，而是为了规则层做稳定、可解释的启发式判断。
    """

    recommended_cues = ("适合", "推荐", "优先", "宜采用", "更适合", "最佳")
    discouraged_cues = ("不适合", "不宜", "避免", "慎用", "不推荐")
    customer_segments = extract_explicit_items(text, ["适用客户", "目标客户", "客户分层", "适用对象"])
    channel_records: Dict[str, ChannelFitRule] = {}

    def extract_channels_after_cue(clause: str, cue_group: Sequence[str], stop_group: Sequence[str]) -> List[str]:
        """在某个提示词之后截取局部短语，并只识别这一段里的渠道。"""

        for cue in cue_group:
            cue_index = clause.find(cue)
            if cue_index >= 0:
                if cue == "适合" and cue_index > 0 and clause[cue_index - 1] == "不":
                    continue
                tail = clause.split(cue, 1)[1]
                for stop_cue in stop_group:
                    if stop_cue in tail:
                        tail = tail.split(stop_cue, 1)[0]
                return detect_channels(tail)
        return []

    def register_channels(channel_names: Sequence[str], fit_score: float, reason: str) -> None:
        """把同一句话中识别到的渠道写入聚合表。"""

        for channel_name in channel_names:
            new_record = ChannelFitRule(
                channel=channel_name,
                fit_score=fit_score,
                customer_segments=list(dict.fromkeys(customer_segments)),
                applicable_scenarios=list(dict.fromkeys(scenarios)),
                reason=reason,
            )
            channel_records[channel_name] = merge_channel_fit_record(channel_records.get(channel_name), new_record)

    for sentence in [item.strip() for item in SENTENCE_SPLIT_PATTERN.split(text) if item.strip()]:
        for clause in [part.strip() for part in re.split(r"[，,；;]", sentence) if part.strip()]:
            recommended_hit = extract_channels_after_cue(clause, recommended_cues, discouraged_cues)
            discouraged_hit = extract_channels_after_cue(clause, discouraged_cues, ())

            # 这里用固定分值而不是自由浮动分值，目的是让 0 代码用户更容易理解。
            # 0.9 表示强适配，0.1 表示强不适配。
            if recommended_hit:
                register_channels(recommended_hit, 0.9, clause)
            if discouraged_hit:
                register_channels(discouraged_hit, 0.1, clause)

    return list(channel_records.values())


def extract_metric_benchmarks(text: str) -> Dict[str, Any]:
    """从文献中抽取常见创业指标的数值、区间和公式基准。"""

    benchmarks: Dict[str, Any] = {}
    records: List[BenchmarkRecord] = []
    for sentence in [item.strip() for item in SENTENCE_SPLIT_PATTERN.split(text) if item.strip()]:
        if not any(cue in normalize_text(sentence) for cue in BENCHMARK_CUES):
            continue
        for clause in [part.strip() for part in re.split(r"[；;]", sentence) if part.strip()]:
            normalized_clause = normalize_text(clause)
            for metric_name, aliases in METRIC_ALIASES.items():
                if not any(alias in normalized_clause for alias in aliases):
                    continue
                alias_pattern = "|".join(re.escape(alias) for alias in aliases)
                range_match = re.search(rf"(?:{alias_pattern}).{{0,12}}?(\d+(?:\.\d+)?)\s*(%|％|元|万元|个月|月|天|倍)?\s*(?:-|~|～|至|到)\s*(\d+(?:\.\d+)?)\s*(%|％|元|万元|个月|月|天|倍)?", clause, re.IGNORECASE)
                single_match = re.search(rf"(?:{alias_pattern}).{{0,12}}?(>=|<=|>|<|不少于|不低于|低于|高于|至少|至多)?\s*(\d+(?:\.\d+)?)\s*(%|％|元|万元|个月|月|天|倍)?", clause, re.IGNORECASE)
                if range_match:
                    record = BenchmarkRecord(metric_name=metric_name, original_text=clause, lower_bound=float(range_match.group(1)), upper_bound=float(range_match.group(3)), unit=range_match.group(2) or range_match.group(4))
                elif single_match:
                    record = BenchmarkRecord(metric_name=metric_name, original_text=clause, comparator=single_match.group(1), value=float(single_match.group(2)), unit=single_match.group(3))
                else:
                    continue
                records.append(record)
                benchmarks.setdefault(metric_name, model_to_dict(record))
    if re.search(r"ltv.{0,10}(>=|>|不低于|至少).{0,5}3\s*[x×\*]\s*cac", normalize_text(text)):
        benchmarks["unit_economics_rule"] = {"expression": "LTV >= 3 * CAC", "original_text": "文献提到了 LTV 与 CAC 的经典单位经济规则。"}
    if records:
        benchmarks["benchmark_items"] = [model_to_dict(record) for record in records]
    return benchmarks


def build_knowledge_card(document: DecodedDocument, output_dir: Path) -> KnowledgeCard:
    """
    把清洗后的文献正文组装成满足系统规范的 KnowledgeCard。

    逻辑原理：
    - 先做文本脱水，抽出标题、摘要、标签、场景、行业、证据来源和行业基准。
    - 再根据抽取结果生成顺序型 `KC_001` 卡片编号。
    - 最后把这些字段对齐到 `KnowledgeCard` 模型，形成系统可直接消费的结构化卡片。
    """

    # 第 1 步：先把正文拆成行，后面抽标题和摘要时会更方便。
    lines = [line for line in document.cleaned_text.splitlines() if line.strip()]

    # 第 2 步：优先读取老师手工写在文中的“标题：xxx”。
    # 如果没有显式标题，就退回到首行或文件名。
    explicit_title = extract_explicit_items(document.cleaned_text, ["标题", "title"])
    title = explicit_title[0] if explicit_title else (lines[0][:80] if lines else Path(document.source_name).stem)

    # 第 3 步：抽取标签、适用场景、所属行业与证据链来源。
    # 这些字段后面会直接被规则层和检索层使用。
    labels = infer_labels(document.cleaned_text)
    scenarios = infer_scenarios(document.cleaned_text, labels)
    industry = infer_industry(document.cleaned_text, labels, scenarios)
    evidence_source = build_evidence_source(document)

    # 第 4 步：抽取指标基准与渠道适配分值。
    # 其中 channel_fit_data 是 H1 规则会重点读取的关键字段。
    industry_benchmarks = extract_metric_benchmarks(document.cleaned_text)
    channel_rules = extract_channel_fit_rules(document.cleaned_text, scenarios)
    if channel_rules:
        industry_benchmarks["channel_fit_data"] = [model_to_dict(rule) for rule in channel_rules]

    # 第 5 步：生成卡片分类、内部编号，以及便于检索的 tags。
    card_type = classify_card_type(document.cleaned_text, labels, scenarios, industry_benchmarks)
    card_id = generate_card_id(output_dir)
    tags = list(dict.fromkeys(([industry] if industry else []) + labels + list(scenarios)))[:12]

    # 第 6 步：把导入过程中的辅助信息写入 metadata。
    # 这些字段主要服务于调试、追溯与后续排错，不直接参与主规则判定。
    metadata = {
        "source_name": document.source_name,
        "source_path": document.source_path,
        "source_suffix": document.suffix,
        "parser": document.parser,
        "encoding_used": document.encoding_used,
        "warnings": document.warnings,
        "source_locator": document.source_locator,
        "evidence_source": evidence_source,
        "source_excerpt": document.cleaned_text[:400],
        "text_length": len(document.cleaned_text),
        "source_fingerprint": stable_identifier(document.source_name, document.cleaned_text),
    }

    # 第 7 步：严格对齐到 Pydantic 模型。
    # 如果某个必填字段缺失，Pydantic 会立刻报错，避免脏数据进入系统。
    return KnowledgeCard(
        id=card_id,
        card_id=card_id,
        type=card_type,
        name=title,
        description=extract_summary(document.cleaned_text, title),
        tags=tags,
        labels=labels,
        applicable_scenarios=scenarios,
        industry=industry,
        evidence_source=evidence_source,
        industry_benchmarks=industry_benchmarks,
        metadata=metadata,
    )


def discover_input_files(input_path: Path) -> DiscoveryResult:
    """
    深度递归扫描输入路径，并区分“待处理文档”和“非文档跳过项”。

    逻辑原理：
    - 用户的 `knowledge_base/raw` 往往会按课程、班级、专题继续分子文件夹存放，因此必须递归到底。
    - 我们只把常见文档格式视为可处理对象，例如 PDF、Word、PPT、TXT、Markdown、HTML。
    - 像图片、压缩包、程序文件等非文档内容不会中断流程，但会被计入“跳过清单”，方便用户核对总数。
    - 扩展名比较统一转成小写，因此 `.PDF`、`.DocX`、`.Md` 这类大小写混合文件也能正常识别。
    """

    document_files: List[Path] = []
    skipped_non_document_files: List[Path] = []

    candidate_paths = [input_path] if input_path.is_file() else list(input_path.rglob("*"))

    for path in candidate_paths:
        if not path.is_file():
            continue

        normalized_name = path.name.lower()
        normalized_suffix = path.suffix.lower()

        # 系统隐藏元数据文件既不是文档，也没有统计价值，直接忽略。
        if normalized_name in IGNORED_FILE_NAMES or path.name.startswith("._"):
            continue

        if normalized_suffix in SUPPORTED_DOCUMENT_SUFFIXES:
            document_files.append(path)
        else:
            skipped_non_document_files.append(path)

    return DiscoveryResult(
        document_files=sorted(document_files),
        skipped_non_document_files=sorted(skipped_non_document_files),
    )


def write_card(card: KnowledgeCard, output_dir: Path) -> Path:
    """
    把知识卡片写入输出目录，并使用便于人工浏览的文件名。

    文件名格式：
    - `KC_行业_类型_序号.json`

    设计原因：
    - 0 代码用户通常直接在资源管理器里找文件，而不是读数据库主键。
    - 因此文件名优先保证“看得懂”，内部唯一性仍由 `card_id` 提供。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / build_output_filename(card)
    output_path.write_text(json.dumps(model_to_dict(card), ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def ingest_source(source_path: Path, output_dir: Path, dry_run: bool) -> Tuple[KnowledgeCard, IngestionResult]:
    """完成单个文件从原始文献到结构化知识卡片的转换流程。"""

    document = load_document(source_path)
    card = build_knowledge_card(document, output_dir)
    output_path = None if dry_run else str(write_card(card, output_dir))
    result = IngestionResult(source_path=str(source_path), card_id=card.card_id or card.id, output_path=output_path, warnings=document.warnings)
    return card, result


def ingest_stdin(stdin_name: str, output_dir: Path, dry_run: bool) -> Tuple[KnowledgeCard, IngestionResult]:
    """支持从标准输入接收原始文献文本，便于快速试跑和调试。"""

    raw_bytes = sys.stdin.buffer.read()
    decoded_text, encoding_used, warnings = decode_bytes_safely(raw_bytes)
    document = DecodedDocument(
        source_name=stdin_name,
        source_path=stdin_name,
        suffix=Path(stdin_name).suffix or ".txt",
        parser="stdin",
        encoding_used=encoding_used,
        cleaned_text=clean_text(decoded_text),
        warnings=warnings,
        source_locator="输入来源: stdin",
    )
    card = build_knowledge_card(document, output_dir)
    output_path = None if dry_run else str(write_card(card, output_dir))
    result = IngestionResult(source_path=stdin_name, card_id=card.card_id or card.id, output_path=output_path, warnings=warnings)
    return card, result


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器，定义脚本的使用方式。"""

    parser = argparse.ArgumentParser(description="把老师提供的原始文献自动转成结构化 KnowledgeCard")
    parser.add_argument("--input", type=str, default=str(DEFAULT_RAW_DIR), help="单个文件或目录路径，默认读取 knowledge_base/raw")
    parser.add_argument("--output", type=str, default=str(DEFAULT_CARDS_DIR), help="输出目录，默认写到 knowledge_base/cards")
    parser.add_argument("--stdin", action="store_true", help="从标准输入读取原始文献")
    parser.add_argument("--stdin-name", type=str, default="stdin.txt", help="标准输入模式下使用的虚拟文件名")
    parser.add_argument("--dry-run", action="store_true", help="只做转换，不落盘")
    parser.add_argument("--stdout", action="store_true", help="把转换后的 JSON 直接打印到标准输出")
    parser.add_argument("--log-file", type=str, default=str(DEFAULT_LOG_FILE), help="导入日志文件，默认写到项目根目录 ingest_log.txt")
    return parser


def main() -> None:
    """执行脚本入口，并根据输入方式批量生成知识卡片。"""

    args = build_arg_parser().parse_args()
    default_raw_dir, default_cards_dir = ensure_knowledge_base_directories()
    output_dir = Path(args.output)
    log_file = Path(args.log_file)
    cards: List[KnowledgeCard] = []
    results: List[IngestionResult] = []
    skipped_count = 0
    failed_count = 0

    if args.stdin:
        # `stdin` 模式只有一份输入，因此把总任务数固定为 1。
        emit_ingest_progress(0, 1, args.stdin_name, 0, skipped_count, failed_count)
        card, result = ingest_stdin(args.stdin_name, output_dir, args.dry_run)
        cards.append(card)
        results.append(result)
        emit_ingest_summary(1, len(cards), skipped_count, failed_count)
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            raise SystemExit(f"输入路径不存在: {input_path}")

        discovery_result = discover_input_files(input_path)
        source_files = discovery_result.document_files
        skipped_non_document_files = discovery_result.skipped_non_document_files
        total_files = len(source_files)

        emit_scan_discovery_summary(total_files, len(skipped_non_document_files))

        for index, source_path in enumerate(source_files, start=1):
            # 在真正处理文件前，先告诉用户当前轮到哪一份资料。
            # 这里展示的是“已完成数量”，因此用 `index - 1`。
            emit_ingest_progress(index - 1, total_files, source_path.name, len(cards), skipped_count, failed_count)
            try:
                card, result = ingest_source(source_path, output_dir, args.dry_run)
                cards.append(card)
                results.append(result)
                for warning_message in result.warnings:
                    append_ingest_log(log_file, "WARNING", str(source_path), warning_message)
            except UnreadableDocumentError as exc:
                append_ingest_log(log_file, "SKIP", str(source_path), str(exc))
                results.append(IngestionResult(source_path=str(source_path), card_id="SKIPPED", output_path=None, warnings=[str(exc)]))
                skipped_count += 1
            except Exception as exc:
                append_ingest_log(log_file, "ERROR", str(source_path), "文件处理失败，已跳过。", {"error": str(exc)})
                results.append(IngestionResult(source_path=str(source_path), card_id="FAILED", output_path=None, warnings=[str(exc)]))
                failed_count += 1

        if total_files > 0:
            emit_ingest_summary(total_files, len(cards), skipped_count, failed_count)

        if input_path == default_raw_dir and not source_files and not skipped_non_document_files:
            print(f"提示：默认原始文献目录为空，请先把文件放入 {default_raw_dir}")
        elif not source_files and skipped_non_document_files:
            print("提示：扫描已完成，但当前目录下没有发现可处理文档，请检查文件格式是否属于 PDF/Word/PPT/TXT/MD/HTML。", file=sys.stderr)

    if args.stdout:
        payload = [model_to_dict(card) for card in cards]
        print(json.dumps(payload[0] if len(payload) == 1 else payload, ensure_ascii=False, indent=2))
        return

    print(f"完成转换: {len(cards)} 张知识卡片")
    for result in results:
        print(json.dumps(model_to_dict(result), ensure_ascii=False))


if __name__ == "__main__":
    main()
