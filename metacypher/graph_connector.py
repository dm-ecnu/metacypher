"""
Simple Neo4j Connection Manager - 极简版
专为论文实验设计，避免频繁建立连接
"""

from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError, ServiceUnavailable
from typing import List, Dict, Any, Optional, Literal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from config import NEO4J_HOST, NEO4J_USER, NEO4J_PASSWORD

# Each graph runs as a separate Neo4j instance on its own bolt port. Host and
# credentials come from config (env-overridable); only the port differs.
CYPHERBENCH_PORTS = {
    "company": 15062,
    "fictional_character": 15063,
    "flight_accident": 15064,
    "geography": 15065,
    "movie": 15066,
    "nba": 15067,
    "politics": 15068,
}

SANDBOX_PORTS = {
    "bloom50": 15091,
    "covid": 15092,
    "er": 15093,
    "gdsc": 15094,
    "healthcare": 15095,
    "legis_graph": 15096,
    "osm": 15097,
    "pole": 15098,
    "twitter_trolls": 15099,
    "wwc": 15100,
}


def _db_config(port: int) -> dict:
    return {
        "uri": f"bolt://{NEO4J_HOST}:{port}",
        "user": NEO4J_USER,
        "password": NEO4J_PASSWORD,
    }


CYPHERBENCH_DBS = {name: _db_config(port) for name, port in CYPHERBENCH_PORTS.items()}
SANDBOX_DBS = {name: _db_config(port) for name, port in SANDBOX_PORTS.items()}

class Neo4jConnectionPool:
    def __init__(self, dataset: str = "cypherbench"):
        self.dataset = dataset
        self.configs = CYPHERBENCH_DBS if dataset == "cypherbench" else SANDBOX_DBS
        self._drivers = {}  # 存储driver对象
        logger.info(f"Initialized pool for {dataset} ({len(self.configs)} databases)")
    
    def connect_all(self):
        for db_name, config in self.configs.items():
            try:
                driver = GraphDatabase.driver(
                    config["uri"],
                    auth=(config["user"], config["password"])
                )
                driver.verify_connectivity()
                self._drivers[db_name] = driver
                logger.info(f"Connected to {db_name}")
            except Exception as e:
                logger.error(f"Failed to connect to {db_name}: {e}")
    
    def execute(self, db_name: str, cypher: str, params: dict = {}) -> List[Dict]:
        if db_name not in self._drivers:
            config = self.configs[db_name]
            driver = GraphDatabase.driver(
                config["uri"],
                auth=(config["user"], config["password"])
            )
            self._drivers[db_name] = driver
        driver = self._drivers[db_name]
        with driver.session(database="neo4j") as session:
            try:
                result = session.run(cypher, params or {})
                return [record.data() for record in result]
            except CypherSyntaxError as e:
                logger.error(f"Syntax error in {db_name}: {e}")
                raise
            except Exception as e:
                logger.error(f"Execution error in {db_name}: {e}")
                raise
    
    def close(self):
        """关闭所有连接"""
        for db_name, driver in self._drivers.items():
            driver.close()
            logger.info(f"Closed {db_name}")
        self._drivers.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_pool(dataset: str = "cypherbench", connect_all: bool = False):
    pool = Neo4jConnectionPool(dataset)
    if connect_all:
        pool.connect_all()
    return pool


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Simple Neo4j Connection Pool Test")
    print("="*70)
    
    # 示例1: 手动管理连接
    print("\n1. Manual connection management:")
    pool = create_pool("cypherbench", connect_all=True)
    
    # 执行多个查询
    result = pool.execute("company", "MATCH (n:Company) RETURN n.name LIMIT 3")
    print(f"Found {len(result)} companies:")
    for r in result:
        print(f"  - {r['n.name']}")
    
    result = pool.execute("movie", "MATCH (n:Movie) RETURN n.name LIMIT 3")
    print(f"\nFound {len(result)} movies:")
    for r in result:
        print(f"  - {r['n.name']}")
    
    pool.close()
    
    # 示例2: 上下文管理器
    print("\n2. Context manager:")
    with create_pool("cypherbench") as pool:
        result = pool.execute("nba", "MATCH (n:Team) RETURN n.name LIMIT 3")
        print(f"Found {len(result)} teams:")
        for r in result:
            print(f"  - {r['n.name']}")
    
    print("\n✅ Test completed!")