"""实时聚类面板，统计各说话人编号和不确定条数。"""
from __future__ import annotations

import sys
from collections import Counter
from typing import Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .streaming_pipeline import StreamingClusterPipeline


def cluster_panel_counts(
    pipe: "StreamingClusterPipeline",
    *,
    use_stable: bool = False,
) -> Tuple[Dict[int, int], int, int]:
    """返回各编号条数、不确定条数和总条数。不确定表示置信度不够没对外输出。"""
    if use_stable:
        vals = pipe.prefix_stable_v_id_for_use_all()
    else:
        vals = pipe.prefix_v_id_for_use_all()
    assigned: Counter[int] = Counter()
    uncertain = 0
    for v in vals:
        if v is None:
            uncertain += 1
        else:
            assigned[int(v)] += 1
    return dict(assigned), uncertain, len(vals)


def format_cluster_panel(
    pipe: "StreamingClusterPipeline",
    *,
    use_stable: bool = False,
    last_key: Optional[str] = None,
) -> str:
    assigned, uncertain, total = cluster_panel_counts(pipe, use_stable=use_stable)
    lines: list[str] = []
    if last_key:
        lines.append(f"最新: {last_key}")
    for vid in sorted(assigned.keys()):
        lines.append(f"v_id {vid}（{assigned[vid]}）")
    lines.append(f"不确定（{uncertain}）")
    lines.append(f"总计（{total}）")
    return "\n".join(lines)


def refresh_cluster_panel(
    pipe: "StreamingClusterPipeline",
    *,
    clear: bool = True,
    use_stable: bool = False,
    last_key: Optional[str] = None,
) -> None:
    text = format_cluster_panel(pipe, use_stable=use_stable, last_key=last_key)
    if clear and sys.stdout.isatty():
        sys.stdout.write("\033[2J\033[H")
    elif clear:
        print("---", flush=True)
    print(text, flush=True)
