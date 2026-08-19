import os

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from neo4j import GraphDatabase


app = FastAPI(
    title="Chenjiaci Neo4j API",
    description="陈家祠知识图谱查询与馆内导航服务",
    version="1.1.0"
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

class EntityQuery(BaseModel):
    name: str


# ============================================================
# 首页
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "Chenjiaci Neo4j API",
        "available_tools": [
            "/health",
            "/entity",
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
# 实体关系
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