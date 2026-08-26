from __future__ import annotations

import os
import re
import unicodedata
from contextlib import asynccontextmanager
from difflib import SequenceMatcher
from typing import Any, Iterable

from fastapi import FastAPI, HTTPException, Query
from neo4j import GraphDatabase
from pydantic import BaseModel, Field


NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def _new_driver():
    if not all((NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)):
        return None
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        connection_timeout=10,
        max_connection_lifetime=1800,
    )


driver = _new_driver()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    if driver is not None:
        driver.close()


app = FastAPI(
    title="Chenjiaci Neo4j API",
    description="陈家祠知识图谱查询、图片实体匹配与馆内导航服务",
    version="2.3.0",
    lifespan=lifespan,
)


class EntityQuery(BaseModel):
    name: str


class VisionQuery(BaseModel):
    candidate_names: list[str] = Field(default_factory=list)


class ImageMatchRequest(BaseModel):
    candidate_names: list[str] = Field(default_factory=list)
    visual_keywords: list[str] = Field(default_factory=list)
    craft: str = ""
    top_k: int = Field(default=5, ge=1, le=10)


def _db():
    if driver is None:
        raise HTTPException(status_code=503, detail="Neo4j 环境变量未配置完整")
    return driver


def normalize(value: Any) -> str:
    """统一大小写、全半角及标点；保留中英文、数字。"""
    text = unicodedata.normalize("NFKC", str(value or "")).lower().strip()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def as_list(value: Any) -> list[str]:
    """兼容 Neo4j 中 list、单字符串以及历史的 |/,/，分隔格式。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        text = str(value).strip().strip("[]")
        items = re.split(r"[|,，;；、]", text)
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        raw = str(item).strip().strip("'\"")
        key = normalize(raw)
        if key and key not in seen:
            result.append(raw)
            seen.add(key)
    return result


def clean_list(values: Iterable[Any]) -> list[str]:
    return as_list(list(values))


def char_bigrams(text: str) -> set[str]:
    text = normalize(text)
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def text_similarity(left: str, right: str) -> float:
    """精确/包含优先，其他情况结合序列相似度与二元字符 Jaccard。"""
    a, b = normalize(left), normalize(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    shorter, longer = sorted((a, b), key=len)
    containment = len(shorter) / len(longer) if shorter in longer else 0.0
    aa, bb = char_bigrams(a), char_bigrams(b)
    jaccard = len(aa & bb) / len(aa | bb) if aa | bb else 0.0
    sequence = SequenceMatcher(None, a, b).ratio()
    return round(max(containment * 0.96, 0.55 * sequence + 0.45 * jaccard), 4)


def _best_match(inputs: list[str], targets: list[str]) -> tuple[float, str | None, str | None]:
    best = (0.0, None, None)
    for source in inputs:
        for target in targets:
            similarity = text_similarity(source, target)
            if similarity > best[0]:
                best = (similarity, source, target)
    return best


def score_entity(
    entity: dict[str, Any], candidate_names: list[str], visual_keywords: list[str], craft: str
) -> dict[str, Any]:
    """纯函数评分，便于离线做标注集回归测试。"""
    name = str(entity.get("name") or "")
    aliases = as_list(entity.get("aliases"))
    labels = list(entity.get("labels") or [])
    entity_keywords = as_list(entity.get("visual_keywords"))
    description = str(entity.get("visual_description") or "")
    recognition_hint = str(entity.get("recognition_hint") or "")
    entity_craft = str(entity.get("craft") or "")

    name_sim, matched_name, matched_name_target = _best_match(candidate_names, [name])
    alias_sim, matched_alias, matched_alias_target = _best_match(candidate_names, aliases)
    # 名称只作为辅助证据，避免 Vision 猜错名称时压过真实视觉特征。
    if name_sim == 1.0:
        name_score, name_match_type = 25.0, "exact_name"
    elif alias_sim == 1.0:
        name_score, name_match_type = 22.0, "exact_alias"
    else:
        best_name_sim = max(name_sim, alias_sim)
        name_score = round(15.0 * best_name_sim, 2) if best_name_sim >= 0.62 else 0.0
        name_match_type = "fuzzy" if name_score else "none"

    matched_keywords: list[dict[str, Any]] = []
    for keyword in visual_keywords:
        # 第一优先：结构化 visual_keywords。
        sim, _, target = _best_match([keyword], entity_keywords)

        # 第二优先：描述和识别提示。短词被描述直接包含时视为强证据。
        keyword_norm = normalize(keyword)
        description_norm = normalize(description)
        hint_norm = normalize(recognition_hint)
        if len(keyword_norm) >= 2 and (
            keyword_norm in description_norm or keyword_norm in hint_norm
        ):
            desc_sim = 1.0
            desc_source = (
                "visual_description"
                if keyword_norm in description_norm
                else "recognition_hint"
            )
        else:
            desc_sim, _, desc_target = _best_match(
                [keyword], [description, recognition_hint]
            )
            if desc_target == description:
                desc_source = "visual_description"
            elif desc_target == recognition_hint:
                desc_source = "recognition_hint"
            else:
                desc_source = "description"

        # 字段化视觉关键词证据强于自然语言描述证据。
        if sim >= 0.58:
            matched_keywords.append({"input": keyword, "target": target, "similarity": sim, "source": "visual_keywords"})
        elif len(keyword_norm) >= 2 and desc_sim >= 0.72:
            matched_keywords.append({"input": keyword, "target": desc_source, "similarity": desc_sim, "source": desc_source})

    keyword_strength = sum(float(x["similarity"]) for x in matched_keywords)
    visual_score = min(48.0, round(keyword_strength * 12.0, 2))
    combo_score = 15.0 if len(matched_keywords) >= 2 else 0.0

    # craft 同时匹配专门字段和视觉关键词，兼容 craft 属性为空的节点。
    craft_targets = [entity_craft] + entity_keywords
    craft_sim, _, matched_craft_target = (
        _best_match([craft], craft_targets) if craft else (0.0, None, None)
    )
    craft_score = 18.0 if craft_sim == 1.0 else (round(14.0 * craft_sim, 2) if craft_sim >= 0.6 else 0.0)

    raw_score = round(name_score + visual_score + combo_score + craft_score, 2)
    has_exact_name = name_match_type in {"exact_name", "exact_alias"}
    penalty = 0.55 if "LowVisualFeature" in labels and not has_exact_name else 1.0
    score = round(raw_score * penalty, 2)
    strong_evidence = (
        len(matched_keywords) >= 2
        or (len(matched_keywords) >= 1 and craft_score > 0)
        or (has_exact_name and len(matched_keywords) >= 1)
        or (has_exact_name and craft_score > 0)
    )

    return {
        **entity,
        "matched_names": [x for x in [matched_name or matched_alias] if x],
        "matched_keywords": matched_keywords,
        "score_details": {
            "name_match_type": name_match_type,
            "name_similarity": max(name_sim, alias_sim),
            "matched_name_target": matched_name_target or matched_alias_target,
            "name_score": name_score,
            "visual_score": visual_score,
            "combo_score": combo_score,
            "craft_similarity": craft_sim,
            "matched_craft_target": matched_craft_target,
            "craft_score": craft_score,
            "low_visual_penalty": penalty,
        },
        "raw_score": raw_score,
        "score": score,
        "strong_evidence": strong_evidence,
    }


def _resolve_synonyms(keywords: list[str]) -> tuple[list[str], dict[str, str]]:
    """支持优化版同义词本体；没有本体或未命中时安全回退。"""
    if not keywords:
        return [], {}
    query = """
    UNWIND $keywords AS raw
    OPTIONAL MATCH (ft:FeatureTerm)-[:HAS_SYNONYM]->(syn:VisualSynonym)
    WHERE toLower(syn.name) = toLower(raw) OR toLower(ft.name) = toLower(raw)
    WITH raw, head(collect(ft.name)) AS canonical
    RETURN raw, coalesce(canonical, raw) AS normalized
    """
    records, _, _ = _db().execute_query(query, keywords=keywords)
    mapping = {str(r["raw"]): str(r["normalized"]) for r in records}
    normalized = clean_list(mapping.get(x, x) for x in keywords)
    return normalized, mapping


@app.get("/")
def root():
    return {"status": "ok", "service": "Chenjiaci Neo4j API", "version": "2.3.0",
            "available_tools": ["/health", "/entity", "/vision-entity", "/image-match", "/indoor-route"]}


@app.get("/health")
def health():
    try:
        _db().verify_connectivity()
        return {"status": "ok", "neo4j": "connected"}
    except HTTPException:
        raise
    except Exception as exc:
        return {"status": "error", "neo4j": "disconnected", "message": str(exc)}


@app.post("/vision-entity")
def vision_entity(data: VisionQuery):
    names = clean_list(data.candidate_names)
    if not names:
        raise HTTPException(status_code=400, detail="candidate_names 不能为空")
    query = """
    MATCH (n) WHERE n.name IN $names OR any(a IN coalesce(n.aliases, []) WHERE a IN $names)
    OPTIONAL MATCH (n)-[r]-(m)
    RETURN n.id AS id, n.name AS name, labels(n) AS labels, properties(n) AS entity,
           collect(DISTINCT {relation:type(r), target:properties(m)}) AS relations
    LIMIT 20
    """
    try:
        records, _, _ = _db().execute_query(query, names=names)
        return {"success": True, "results": [r.data() for r in records]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"知识图谱查询失败：{exc}") from exc


@app.post("/entity")
def search_entity(data: EntityQuery):
    name = data.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="实体名称不能为空")
    query = """
    MATCH (n)
    WHERE toLower(n.name) CONTAINS toLower($name)
       OR any(a IN coalesce(n.aliases, []) WHERE toLower(a) CONTAINS toLower($name))
       OR toString(n.id) CONTAINS $name
    OPTIONAL MATCH (n)-[r]-(m)
    RETURN labels(n) AS node_type, properties(n) AS entity, type(r) AS relation,
           labels(m) AS related_type, properties(m) AS related_entity
    ORDER BY CASE WHEN toLower(n.name)=toLower($name) THEN 0 ELSE 1 END
    LIMIT 30
    """
    try:
        records, _, _ = _db().execute_query(query, name=name)
        results = [r.data() for r in records]
        return {"success": True, "query": name, "count": len(results), "results": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"知识图谱查询失败：{exc}") from exc


@app.post("/image-match")
def image_match(data: ImageMatchRequest):
    candidate_names = clean_list(data.candidate_names)
    visual_keywords = clean_list(data.visual_keywords)
    craft = data.craft.strip()
    input_data = {"candidate_names": candidate_names, "visual_keywords": visual_keywords, "craft": craft}
    if not candidate_names and not visual_keywords and not craft:
        return {"success": False, "matched": False, "input": input_data,
                "message": "图片未提取到有效视觉特征。", "results": []}

    try:
        normalized_keywords, synonym_mapping = _resolve_synonyms(visual_keywords)
        query = """
        MATCH (n:ImageRecognizable)
        WHERE NOT 'ScenicSpot' IN labels(n)
          AND (
              (n.visual_keywords IS NOT NULL AND size(n.visual_keywords) > 0)
              OR trim(coalesce(n.visual_description, '')) <> ''
              OR trim(coalesce(n.recognition_hint, '')) <> ''
          )
        OPTIONAL MATCH (n)-[:LOCATED_IN|DISPLAYED_IN|EXHIBIT_DISPLAYED_IN_SPACE]->(loc)
        RETURN n.id AS id, n.name AS name, labels(n) AS labels,
               n.aliases AS aliases, n.craft AS craft,
               n.visual_keywords AS visual_keywords,
               n.visual_description AS visual_description,
               n.recognition_hint AS recognition_hint,
               collect(DISTINCT loc.name) AS related_locations
        """
        records, _, _ = _db().execute_query(query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"图片实体匹配失败：{exc}") from exc

    ranked = [score_entity(r.data(), candidate_names, normalized_keywords, craft) for r in records]
    ranked = [x for x in ranked if x["score"] > 0]
    # 不设置绝对分数或动态阈值：只要获得正分，就按总分决定胜者。
    # 同分时优先视觉证据、组合特征、工艺，最后才看名称。
    ranked.sort(
        key=lambda x: (
            x["score"],
            x["score_details"]["visual_score"],
            x["score_details"]["combo_score"],
            x["score_details"]["craft_score"],
            x["score_details"]["name_score"],
        ),
        reverse=True,
    )
    results = ranked[: data.top_k]
    for item in results:
        item.pop("strong_evidence", None)

    if not results:
        return {"success": True, "matched": False, "input": input_data,
                "normalized_keywords": normalized_keywords, "synonym_mapping": synonym_mapping,
                "message": "所有候选实体得分均为 0，请补充至少一个可匹配特征。", "results": []}

    gap = round(results[0]["score"] - (results[1]["score"] if len(results) > 1 else 0.0), 2)
    return {"success": True, "matched": True, "input": input_data,
            "normalized_keywords": normalized_keywords, "synonym_mapping": synonym_mapping,
            "best_match": results[0], "score_gap": gap, "count": len(results), "results": results}


@app.get("/indoor-route")
def get_indoor_route(start: str = Query(...), end: str = Query(...)):
    start, end = start.strip(), end.strip()
    if not start or not end:
        raise HTTPException(status_code=400, detail="起点和终点不能为空")
    if start == end:
        return {"success": True, "found": True, "start": start, "end": end,
                "route": [start], "route_text": start, "steps": 0, "message": "您已经在目标地点。"}
    query = """
    MATCH (start:Space {name:$start}), (end:Space {name:$end})
    OPTIONAL MATCH p=shortestPath((start)-[:NEXT_STOP|ADJACENT_TO*..20]-(end))
    RETURN start.name AS start_name, end.name AS end_name,
           CASE WHEN p IS NULL THEN [] ELSE [n IN nodes(p)|n.name] END AS route,
           CASE WHEN p IS NULL THEN [] ELSE [r IN relationships(p)|type(r)] END AS relation_types,
           CASE WHEN p IS NULL THEN 0 ELSE length(p) END AS steps
    """
    try:
        records, _, _ = _db().execute_query(query, start=start, end=end)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"馆内路线查询失败：{exc}") from exc
    if not records or not records[0]["route"]:
        return {"success": True, "found": False, "start": start, "end": end, "route": [],
                "message": "当前导航数据中未找到起点、终点或可用路径。"}
    record = records[0]
    return {"success": True, "found": True, "start": record["start_name"], "end": record["end_name"],
            "route": record["route"], "route_text": " → ".join(record["route"]),
            "steps": record["steps"], "relation_types": record["relation_types"],
            "source": "陈家祠馆内导航知识图谱"}
