"""
简化版 Neo4j 客户端

支持CypherBench的11个数据库连接和查询，内置所有配置，保留重试逻辑。

使用示例:
    # 单数据库连接
    connector = Neo4jConnector("art")
    result = connector.execute("MATCH (n:Painting) RETURN n.name LIMIT 5")
    connector.close()
    
    # 多数据库管理
    manager = Neo4jConnectionManager()
    art_conn = manager.get_connector("art")
    bio_conn = manager.get_connector("biology")
    manager.close_all()
    
    # With语句
    with Neo4jConnector("art") as conn:
        result = conn.execute("MATCH (n) RETURN n LIMIT 10")
"""

import time
from typing import Any, Dict, List, Optional, cast, LiteralString, Tuple
from neo4j import GraphDatabase, Record
from neo4j.exceptions import (
    ServiceUnavailable,
    SessionExpired,
    TransientError,
    CypherSyntaxError,
    ClientError
)


# ============= 内置配置 =============

from config import NEO4J_HOST, NEO4J_USER, NEO4J_PASSWORD

# Each CypherBench graph runs as a separate Neo4j instance on its own bolt
# port. Host and credentials are env-overridable via config; only the port
# differs per graph.
_DATABASE_PORTS = {
    "art": 15060,
    "biology": 15061,
    "company": 15062,
    "fictional_character": 15063,
    "flight_accident": 15064,
    "geography": 15065,
    "movie": 15066,
    "nba": 15067,
    "politics": 15068,
    "soccer": 15069,
    "terrorist_attack": 15070,
}

DATABASE_CONFIG = {
    name: {
        "host": NEO4J_HOST,
        "port": port,
        "username": NEO4J_USER,
        "password": NEO4J_PASSWORD,
        "database": "neo4j",
    }
    for name, port in _DATABASE_PORTS.items()
}

DEFAULT_RETRY_CONFIG = {
    "max_retries": 3,
    "retry_delay": 2,
    "exponential_backoff": True
}

DEFAULT_QUERY_TIMEOUT = 30


# ============= 异常类 =============

class Neo4jConnectionError(Exception):
    """Neo4j 连接错误"""
    pass


class Neo4jQueryError(Exception):
    """Neo4j 查询错误"""
    pass


# ============= 单数据库连接器 =============

class Neo4jConnector:
    """
    单个 Neo4j 数据库的连接器
    
    支持:
    - 查询执行（带自动重试）
    - 连接测试
    - 上下文管理器
    
    示例:
        >>> connector = Neo4jConnector("art")
        >>> result = connector.execute("MATCH (n:Painting) RETURN n.name LIMIT 5")
        >>> print(len(result))
        5
        >>> connector.close()
        
        或使用with语句:
        >>> with Neo4jConnector("art") as conn:
        ...     result = conn.execute("MATCH (n) RETURN count(n)")
    """
    # 进程内复用 driver，避免重复建立连接
    _DRIVER_CACHE: Dict[Tuple[str, str, str], Any] = {}

    def __init__(
        self,
        database_name: str,
        max_retries: int = 3,
        retry_delay: int = 2,
        exponential_backoff: bool = True,
        timeout: int = 30
    ):
        """
        初始化 Neo4j 连接器
        
        参数:
            database_name: 数据库名称 (art, biology, company, etc.)
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            exponential_backoff: 是否使用指数退避
            timeout: 查询超时时间（秒）
        """
        if database_name not in DATABASE_CONFIG:
            available = list(DATABASE_CONFIG.keys())
            raise ValueError(
                f"Unknown database '{database_name}'. "
                f"Available: {available}"
            )
        
        self.database_name = database_name
        config = DATABASE_CONFIG[database_name]
        
        self.host = config["host"]
        self.port = config["port"]
        self.username = config["username"]
        self.password = config["password"]
        self.database = config["database"]
        
        # 重试配置
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.exponential_backoff = exponential_backoff
        self.timeout = timeout
        
        # 创建连接（进程内复用，避免重复建立）
        self.uri = f"bolt://{self.host}:{self.port}"
        self._driver = None
        self._ensure_driver()

    def _cache_key(self) -> Tuple[str, str, str]:
        return (self.uri, self.username, self.password)

    def _ensure_driver(self) -> None:
        if self._driver is not None:
            return
        key = self._cache_key()
        cached = self._DRIVER_CACHE.get(key)
        if cached is not None:
            self._driver = cached
            return
        self._connect()
        # _connect 成功后 self._driver 一定非空
        self._DRIVER_CACHE[key] = self._driver

    def _invalidate_driver(self) -> None:
        # 不主动 close 缓存 driver，避免影响其他复用者；仅让本实例下次重建/复用
        self._driver = None
    
    def _connect(self):
        """建立连接"""
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
        except Exception as e:
            raise Neo4jConnectionError(f"Failed to connect to {self.uri}: {e}")
    
    def test_connection(self) -> bool:
        """
        测试连接是否正常

        返回:
            bool: 连接是否成功
        """
        try:
            resp = self.execute("RETURN 1 AS test")
            return bool(resp.get("ok")) and resp.get("records") and resp["records"][0].get("test") == 1
        except Exception:
            return False
    
    def execute(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        执行 Cypher 查询（带重试）

        关键行为:
        - 不输出日志/print
        - 无论成功还是来自 Neo4j 的执行报错信息，都返回结构化反馈
        - 仅对连接/会话/临时错误做重试

        返回:
            Dict[str, Any]:
                {
                  "ok": bool,
                  "records": List[Dict[str, Any]],
                  "error": {"type": str, "message": str, "code": Optional[str]} | None,
                  "meta": {"attempts": int, "elapsed_sec": float}
                }
        """
        self._ensure_driver()

        timeout = timeout or self.timeout
        start_all = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                assert self._driver is not None
                query = cast(LiteralString, cypher)
                with self._driver.session(database=self.database) as session:
                    result = session.run(query, parameters or {}, timeout=timeout)
                    records = list(result)

                rows = [r.data() for r in records]
                return {
                    "ok": True,
                    "records": rows,
                    "error": None,
                    "meta": {
                        "attempts": attempt + 1,
                        "elapsed_sec": time.time() - start_all,
                    },
                }

            except CypherSyntaxError as e:
                # 语法错误不重试，直接返回
                return {
                    "ok": False,
                    "records": [],
                    "error": {
                        "type": "CypherSyntaxError",
                        "message": str(e),
                        "code": getattr(e, "code", None),
                    },
                    "meta": {
                        "attempts": attempt + 1,
                        "elapsed_sec": time.time() - start_all,
                    },
                }

            except ClientError as e:
                # 客户端错误（如约束/权限/类型等）不重试，直接返回
                return {
                    "ok": False,
                    "records": [],
                    "error": {
                        "type": "ClientError",
                        "message": str(e),
                        "code": getattr(e, "code", None),
                    },
                    "meta": {
                        "attempts": attempt + 1,
                        "elapsed_sec": time.time() - start_all,
                    },
                }

            except (ServiceUnavailable, SessionExpired, TransientError) as e:
                # 连接或临时错误，可重试；先让 driver 失效以触发重连/复用
                self._invalidate_driver()

                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt if self.exponential_backoff else 1)
                    time.sleep(delay)
                    self._ensure_driver()
                    continue

                return {
                    "ok": False,
                    "records": [],
                    "error": {
                        "type": e.__class__.__name__,
                        "message": str(e),
                        "code": getattr(e, "code", None),
                    },
                    "meta": {
                        "attempts": attempt + 1,
                        "elapsed_sec": time.time() - start_all,
                    },
                }

            except Exception as e:
                # 其他错误：保持原有“可重试”的策略，但不打印；最终返回错误信息
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt if self.exponential_backoff else 1)
                    time.sleep(delay)
                    continue

                return {
                    "ok": False,
                    "records": [],
                    "error": {
                        "type": e.__class__.__name__,
                        "message": str(e),
                        "code": getattr(e, "code", None),
                    },
                    "meta": {
                        "attempts": attempt + 1,
                        "elapsed_sec": time.time() - start_all,
                    },
                }

        return {
            "ok": False,
            "records": [],
            "error": {
                "type": "Neo4jQueryError",
                "message": "Unexpected execution path",
                "code": None,
            },
            "meta": {
                "attempts": self.max_retries + 1,
                "elapsed_sec": time.time() - start_all,
            },
        }
    
    def execute_read(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """执行只读查询（用于 MATCH），等同于 execute()"""
        return self.execute(cypher, parameters)
    
    def execute_write(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """执行写操作查询（用于 CREATE/UPDATE/DELETE），等同于 execute()"""
        return self.execute(cypher, parameters)
    
    def close(self):
        """关闭连接"""
        if self._driver:
            self._driver.close()
            self._driver = None
    
    def __enter__(self):
        """支持 with 语句"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时不自动关闭连接（便于同一进程内复用）"""
        return False
    
    def __del__(self):
        """析构时关闭连接"""
        self.close()
    
    def __repr__(self):
        return f"Neo4jConnector(database='{self.database_name}', uri='{self.uri}')"


# ============= 多数据库连接管理器 =============

class Neo4jConnectionManager:
    """
    多数据库连接管理器
    
    支持:
    - 懒加载连接（第一次使用时创建）
    - 统一管理多个数据库连接
    - 批量关闭连接
    
    示例:
        >>> manager = Neo4jConnectionManager()
        >>> art_conn = manager.get_connector("art")
        >>> result = art_conn.execute("MATCH (n) RETURN n LIMIT 5")
        >>> manager.close_all()
        
        或使用with语句:
        >>> with Neo4jConnectionManager() as manager:
        ...     art_conn = manager.get_connector("art")
        ...     bio_conn = manager.get_connector("biology")
    """
    
    def __init__(self):
        """初始化连接管理器"""
        self.connectors: Dict[str, Neo4jConnector] = {}
    
    def get_connector(self, database_name: str) -> Neo4jConnector:
        """
        获取指定数据库的连接器（懒加载）
        
        参数:
            database_name: 数据库名称（art, biology, company, etc.）
        
        返回:
            Neo4jConnector: 数据库连接器
        
        异常:
            ValueError: 数据库配置不存在
            Neo4jConnectionError: 连接失败
        """
        # 如果连接器已存在，直接返回
        if database_name in self.connectors:
            return self.connectors[database_name]
        
        # 创建新连接器
        try:
            connector = Neo4jConnector(database_name)
            
            # 测试连接
            if not connector.test_connection():
                raise Neo4jConnectionError(
                    f"Connection test failed for database: {database_name}"
                )
            
            # 存储连接器
            self.connectors[database_name] = connector
            
            return connector
        
        except Exception as e:
            raise Neo4jConnectionError(f"Failed to create connector for {database_name}: {e}")
    
    def close_all(self):
        """关闭所有连接"""
        for connector in self.connectors.values():
            try:
                connector.close()
            except Exception:
                pass
        
        self.connectors.clear()
    
    def list_databases(self) -> List[str]:
        """
        列出配置中的所有数据库
        
        返回:
            List[str]: 数据库名称列表
        """
        return list(DATABASE_CONFIG.keys())
    
    def list_active_connections(self) -> List[str]:
        """
        列出当前活跃的连接
        
        返回:
            List[str]: 活跃连接的数据库名称列表
        """
        return list(self.connectors.keys())
    
    def __enter__(self):
        """支持 with 语句"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时不自动关闭连接（便于同一进程内复用）"""
        return False
    
    def __del__(self):
        """析构时关闭所有连接"""
        self.close_all()
    
    def __repr__(self):
        active = len(self.connectors)
        total = len(DATABASE_CONFIG)
        return f"Neo4jConnectionManager(active={active}, total={total})"


# ============= 使用示例 =============

if __name__ == "__main__":
    print("=== Neo4j Client 使用示例 ===\n")
    
    # 示例1: 单数据库连接
    print("示例1: 单数据库连接")
    print("-" * 50)
    
    try:
        with Neo4jConnector("art") as conn:
            # 测试连接
            print(f"✓ 连接成功: {conn}")
            
            # 执行查询
            resp = conn.execute("MATCH (n:Painting) RETURN n.name LIMIT 5")
            if not resp["ok"]:
                raise RuntimeError(resp["error"])
            result = resp["records"]
            print(f"✓ 查询结果: {len(result)} 条记录")
            
            # 显示结果
            for i, row in enumerate(result, 1):
                print(f"  {i}. {row.get('n.name')}")
    
    except Exception as e:
        print(f"✗ 错误: {e}")
    
    print()
    
    # 示例2: 多数据库管理
    print("示例2: 多数据库管理")
    print("-" * 50)
    
    try:
        with Neo4jConnectionManager() as manager:
            # 列出所有数据库
            databases = manager.list_databases()
            print(f"✓ 可用数据库: {databases[:3]}... (共{len(databases)}个)")
            
            # 连接到多个数据库
            art_conn = manager.get_connector("art")
            bio_conn = manager.get_connector("biology")
            
            print(f"✓ 活跃连接: {manager.list_active_connections()}")
            
            # 执行查询
            art_resp = art_conn.execute("MATCH (n) RETURN count(n) as total")
            bio_resp = bio_conn.execute("MATCH (n) RETURN count(n) as total")
            if not art_resp["ok"]:
                raise RuntimeError(art_resp["error"])
            if not bio_resp["ok"]:
                raise RuntimeError(bio_resp["error"])
            art_result = art_resp["records"]
            bio_result = bio_resp["records"]
            
            print(f"✓ Art数据库节点总数: {art_result[0]['total']}")
            print(f"✓ Biology数据库节点总数: {bio_result[0]['total']}")
    
    except Exception as e:
        print(f"✗ 错误: {e}")
    
    print()
    
    # 示例3: 重试机制演示
    print("示例3: 参数化查询")
    print("-" * 50)
    
    try:
        with Neo4jConnector("company") as conn:
            # 参数化查询
            resp = conn.execute(
                "MATCH (n:Company) WHERE n.name CONTAINS $keyword RETURN n.name LIMIT 3",
                parameters={"keyword": "Tech"}
            )
            if not resp["ok"]:
                raise RuntimeError(resp["error"])
            result = resp["records"]

            print(f"✓ 找到 {len(result)} 家公司包含'Tech':")
            for row in result:
                print(f"  - {row.get('n.name')}")
    
    except Exception as e:
        print(f"✗ 错误: {e}")
    
    print()
    
    # 示例4: 批量处理
    print("示例4: 批量处理多个数据库")
    print("-" * 50)
    
    try:
        manager = Neo4jConnectionManager()
        
        target_dbs = ["art", "biology", "company"]
        
        for db_name in target_dbs:
            conn = manager.get_connector(db_name)
            resp = conn.execute("MATCH (n) RETURN count(n) as total")
            if not resp["ok"]:
                raise RuntimeError(resp["error"])
            total = resp["records"][0]["total"]
            print(f"✓ {db_name:20s}: {total:>10,} nodes")
        
        manager.close_all()
    
    except Exception as e:
        print(f"✗ 错误: {e}")
    
    print("\n=== 所有示例完成 ===")