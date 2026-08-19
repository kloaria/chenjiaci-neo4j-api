import os

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from typing import List
from neo4j import GraphDatabase


app = FastAPI(
    title="Chenjiaci Neo4j API",
    description="陈家祠知识图谱查询、图片识别与馆内导航服务",
    version="1.2.0"
)


# ============================================================
# Aura
# ============================================================

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)


# ============================================================
# Model
# ============================================================

# 原来的文字实体查询
class EntityQuery(BaseModel):
    name: str


# 新增：图片识别匹配
class ImageMatchRequest(BaseModel):
    candidate_names: List[str] = Field(default_factory=list)
    visual_keywords: List[str] = Field(default_factory=list)
    craft: str = ""


# ============================================================
# 首页
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "Chenjiaci Neo4j API",
        "version": "1.2.0",
        "available_tools": [
            "/health",
            "/entity",
            "/image-match",
            "/indoor-route"
        ]
    }


# ============================================================
# Aura 健康检查
# ============================================================

@app.get("/health")
def health():

    try:

        driver.verify_connectivity()

        return {
            "status": "ok",
            "neo4j": "connected"
        }

    except Exception as e:

        return {
            "status": "error",
            "neo4j": "disconnected",
            "message": str(e)
        }


# ============================================================
# 实体关系查询
# 原功能：文字问题使用
# ============================================================

@app.post("/entity")
def search_entity(data: EntityQuery):

    name = data.name.strip()

    if not name:

        raise HTTPException(
            status_code=400,
            detail="实体名称不能为空"
        )

    query = """
    MATCH (n)
    WHERE n.name CONTAINS $name
       OR toString(n.id) CONTAINS $name

    OPTIONAL MATCH (n)-[r]-(m)

    RETURN
        labels(n) AS node_type,
        properties(n) AS entity,
        type(r) AS relation,
        labels(m) AS related_type,
        properties(m) AS related_entity

    ORDER BY
        CASE
            WHEN n.name = $name THEN 0
            ELSE 1
        END

    LIMIT 30
    """

    try:

        records, summary, keys = driver.execute_query(
            query,
            name=name
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"知识图谱查询失败：{str(e)}"
        )

    results = [
        record.data()
        for record in records
    ]

    return {
        "success": True,
        "query": name,
        "count": len(results),
        "results": results
    }


# ============================================================
# 图片识别实体匹配
# 新功能：Dify Vision 使用
# ============================================================

@app.post("/image-match")
def image_match(data: ImageMatchRequest):

    # --------------------------------------------------------
    # 1. 清洗 Vision LLM 传过来的数据
    # --------------------------------------------------------

    candidate_names = [
        x.strip()
        for x in data.candidate_names
        if x and x.strip()
    ]

    visual_keywords = [
        x.strip()
        for x in data.visual_keywords
        if x and x.strip()
    ]

    craft = data.craft.strip() if data.craft else ""

    # 如果 Vision 什么都没有识别出来
    if not candidate_names and not visual_keywords and not craft:

        raise HTTPException(
            status_code=400,
            detail="candidate_names、visual_keywords 和 craft 不能同时为空"
        )


    # --------------------------------------------------------
    # 2. Neo4j 图片实体匹配
    #
    # 评分：
    # name 精确命中          +100
    # alias 每命中一个        +40
    # craft 匹配             +30
    # visual keyword 每个     +10
    # --------------------------------------------------------

    query = """
    WITH
        $candidate_names AS candidate_names,
        $visual_keywords AS visual_keywords,
        trim(coalesce($craft, '')) AS craft

    MATCH (n:ImageRecognizable)

    WITH
        n,
        candidate_names,
        visual_keywords,
        craft,

        CASE
            WHEN n.name IN candidate_names
            THEN 100
            ELSE 0
        END AS name_score,

        [
            x IN candidate_names
            WHERE x IN coalesce(n.aliases, [])
        ] AS matched_aliases,

        [
            x IN visual_keywords
            WHERE x IN coalesce(n.visual_keywords, [])
        ] AS matched_keywords,

        CASE
            WHEN craft <> ''
                 AND (
                     craft = coalesce(n.craft, '')
                     OR craft IN coalesce(n.visual_keywords, [])
                 )
            THEN 30
            ELSE 0
        END AS craft_score


    WITH
        n,
        matched_aliases,
        matched_keywords,

        name_score,

        size(matched_aliases) * 40
            AS alias_score,

        size(matched_keywords) * 10
            AS visual_score,

        craft_score


    WITH
        n,
        matched_aliases,
        matched_keywords,

        name_score,
        alias_score,
        visual_score,
        craft_score,

        name_score
        + alias_score
        + visual_score
        + craft_score
        AS score


    WHERE score >= 40


    # --------------------------------------------------------
    # 3. 找到实体后，继续利用原图谱查询位置
    # --------------------------------------------------------

    OPTIONAL MATCH
        (n)-[:LOCATED_IN|DISPLAYED_IN|EXHIBIT_DISPLAYED_IN_SPACE]->(loc)


    RETURN
        n.id AS id,
        n.name AS name,

        labels(n) AS labels,

        n.craft AS craft,

        n.location AS location,

        collect(DISTINCT loc.name) AS related_locations,

        n.visual_keywords AS visual_keywords,

        n.visual_description AS visual_description,

        n.recognition_hint AS recognition_hint,

        matched_aliases,

        matched_keywords,

        name_score,
        alias_score,
        visual_score,
        craft_score,

        score

    ORDER BY score DESC

    LIMIT 5
    """


    # --------------------------------------------------------
    # 4. 执行 Neo4j 查询
    # --------------------------------------------------------

    try:

        records, summary, keys = driver.execute_query(
            query,
            candidate_names=candidate_names,
            visual_keywords=visual_keywords,
            craft=craft
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"图片实体匹配失败：{str(e)}"
        )


    results = [
        record.data()
        for record in records
    ]


    # --------------------------------------------------------
    # 5. 没找到
    # --------------------------------------------------------

    if not results:

        return {
            "success": True,
            "matched": False,

            "input": {
                "candidate_names": candidate_names,
                "visual_keywords": visual_keywords,
                "craft": craft
            },

            "message": "当前知识图谱中没有找到置信度足够高的图片匹配实体。",

            "results": []
        }


    # --------------------------------------------------------
    # 6. 找到了
    # --------------------------------------------------------

    best_match = results[0]

    return {
        "success": True,
        "matched": True,

        "input": {
            "candidate_names": candidate_names,
            "visual_keywords": visual_keywords,
            "craft": craft
        },

        "best_match": best_match,

        "count": len(results),

        "results": results
    }


# ============================================================
# 馆内导航
# ============================================================

@app.get("/indoor-route")
def get_indoor_route(

    start: str = Query(
        ...,
        description="起点，例如：前院、聚贤堂、祖堂"
    ),

    end: str = Query(
        ...,
        description="终点，例如：聚贤堂、祖堂"
    )
):

    start = start.strip()
    end = end.strip()

    if not start or not end:

        raise HTTPException(
            status_code=400,
            detail="起点和终点不能为空"
        )

    if start == end:

        return {
            "success": True,
            "found": True,
            "start": start,
            "end": end,
            "route": [start],
            "route_text": start,
            "steps": 0,
            "distance": None,
            "estimated_time": None,
            "message": "您已经在目标地点。",
            "source": "陈家祠馆内导航知识图谱"
        }


    query = """
    MATCH (start:Space {name: $start})
    MATCH (end:Space {name: $end})

    OPTIONAL MATCH p = shortestPath(
        (start)-[:NEXT_STOP|ADJACENT_TO*..20]-(end)
    )

    RETURN
        start.name AS start_name,
        end.name AS end_name,

        CASE
            WHEN p IS NULL THEN []
            ELSE [n IN nodes(p) | n.name]
        END AS route,

        CASE
            WHEN p IS NULL THEN []
            ELSE [r IN relationships(p) | type(r)]
        END AS relation_types,

        CASE
            WHEN p IS NULL THEN 0
            ELSE length(p)
        END AS steps
    """


    try:

        records, summary, keys = driver.execute_query(
            query,
            start=start,
            end=end
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"馆内路线查询失败：{str(e)}"
        )


    if not records:

        return {
            "success": True,
            "found": False,
            "start": start,
            "end": end,
            "route": [],
            "message": "当前数据库中未找到起点或终点。"
        }


    record = records[0]

    route = record["route"]


    if not route:

        return {
            "success": True,
            "found": False,
            "start": start,
            "end": end,
            "route": [],
            "message": "当前导航数据中暂未找到两个地点之间的路径。"
        }


    return {
        "success": True,
        "found": True,

        "start": record["start_name"],
        "end": record["end_name"],

        "route": route,

        "route_text": " → ".join(route),

        "steps": record["steps"],

        "relation_types": record["relation_types"],

        "distance": None,
        "estimated_time": None,

        "source": "陈家祠馆内导航知识图谱"
    }