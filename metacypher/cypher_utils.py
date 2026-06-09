"""
Cypher query helper functions.

Extracted from subgraph_retrieval.py: value formatting, date parsing,
numeric parsing, and MATCH clause construction utilities.
"""

import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _format_cypher_value(value: Any) -> str:
    """格式化Cypher值，字符串使用双引号以避免单引号冲突"""
    if isinstance(value, str):
        escaped = value.replace('\\', '\\\\').replace('"', '\\"')
        return f"\"{escaped}\""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "NULL"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_cypher_value(v) for v in value) + "]"
    return str(value)


def _format_cypher_date(value: Any) -> Optional[str]:
    """将输入转换为Cypher date(...) 表达式；若无法解析则返回None。"""
    if value is None:
        return None
    if isinstance(value, datetime.date):
        return f"date({_format_cypher_value(value.isoformat())})"
    if isinstance(value, (datetime.datetime, datetime.time)):
        try:
            return f"date({_format_cypher_value(value.date().isoformat())})"
        except Exception:
            return None
    if isinstance(value, int):
        return f"date({_format_cypher_value(f'{value:04d}-01-01')})"
    if isinstance(value, str):
        text = value.strip()
        for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%Y"):
            try:
                if pattern == "%Y":
                    parsed = datetime.date(int(text), 1, 1)
                else:
                    parsed = datetime.datetime.strptime(text, pattern).date()
                return f"date({_format_cypher_value(parsed.isoformat())})"
            except (ValueError, TypeError):
                continue
        logger.warning("[CypherDate] Unable to parse date literal '%s'; skipping constraint", text)
        return None
    return None


def _parse_numeric_value(value: Any, allow_float: bool = False) -> Optional[float]:
    """尝试解析数值类型，若失败返回None"""
    if value is None:
        return None
    try:
        if allow_float:
            return float(value)
        return int(value)
    except (ValueError, TypeError):
        logger.warning("[Numeric] 无法解析数值 '%s' (allow_float=%s)", value, allow_float)
        return None


def build_match_clause_and_var_map(path: Any) -> Tuple[str, Dict[str, str]]:
    """根据PathInstance构建MATCH子句并返回node_id到变量名的映射"""
    parts: List[str] = []
    node_var_map: Dict[str, str] = {}
    for i, node in enumerate(path.nodes):
        node_label = node.get('label', 'Node')
        node_props = node.get('properties', {}) or {}
        var = f"n{i}"
        node_id = node.get('id')
        if node_id is not None:
            node_var_map[node_id] = var
        if node_props:
            prop_strs = [
                f"{key}: {_format_cypher_value(value)}"
                for key, value in node_props.items()
            ]
            parts.append(f"({var}:{node_label} {{{', '.join(prop_strs)}}})")
        else:
            parts.append(f"({var}:{node_label})")
        if i < len(path.edges):
            edge = path.edges[i]
            rel_type = edge.get('rel_type') or 'REL'
            rel_var = f"r{i}"
            from_id = edge.get('from')
            to_id = edge.get('to')
            curr_id = node.get('id')
            next_id = path.nodes[i + 1].get('id') if (i + 1) < len(path.nodes) else None
            if from_id and to_id and curr_id and next_id:
                if curr_id == from_id and next_id == to_id:
                    parts.append(f"-[{rel_var}:{rel_type}]->")
                elif curr_id == to_id and next_id == from_id:
                    parts.append(f"<-[{rel_var}:{rel_type}]-")
                else:
                    parts.append(f"-[{rel_var}:{rel_type}]-")
            else:
                parts.append(f"-[{rel_var}:{rel_type}]-")
    return "MATCH " + "".join(parts), node_var_map


def _extract_match_where(query: str) -> str:
    """提取Cypher查询的MATCH/WHERE部分"""
    if not query:
        return ""
    upper_q = query.upper()
    idx = upper_q.find("RETURN")
    if idx != -1:
        return query[:idx].strip()
    return query.strip()
