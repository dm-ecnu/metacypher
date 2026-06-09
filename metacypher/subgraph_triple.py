"""
子图三元组数据结构

版本: 2.0
日期: 2025-12-31
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from copy import deepcopy
import networkx as nx
import numpy as np


def _build_pattern_structure_from_path(path_instance: Any) -> Dict[str, Any]:
    node_labels = [node.get('label') for node in path_instance.nodes if node.get('label')]
    rel_types = [edge.get('rel_type') for edge in path_instance.edges if edge.get('rel_type')]
    return {
        'node_labels': node_labels,
        'rel_types': rel_types,
        'num_hops': len(path_instance.edges),
        'num_nodes': len(path_instance.nodes),
        'num_edges': len(path_instance.edges),
        'has_properties': any(node.get('properties') for node in path_instance.nodes),
        'is_optional': False
    }


@dataclass
class SubgraphTriple:
    """子图三元组: (子图实例, 路径模式, 语义描述)"""
    
    subgraph: nx.MultiDiGraph
    sample_paths: List[Dict[str, Any]]
    cypher_pattern: str
    pattern_structure: Dict[str, Any] = field(default_factory=dict)
    semantic_description: str = ""
    embedding: Optional[np.ndarray] = None
    pattern_signature: str = ""
    sample_paths_total_hits: int = 0
    
    triple_id: str = ""
    pattern_id: str = ""
    anchor_nodes: List[str] = field(default_factory=list)
    hop_count: int = 1
    score: float = 0.0
    parent_id: Optional[str] = None
    is_optional: bool = False
    
    num_nodes: int = 0
    num_edges: int = 0
    num_sample_paths: int = 0
    debug_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.subgraph:
            self.num_nodes = self.subgraph.number_of_nodes()
            self.num_edges = self.subgraph.number_of_edges()
        if self.sample_paths:
            self.num_sample_paths = len(self.sample_paths)
        if self.cypher_pattern and not self.pattern_structure:
            self.pattern_structure = self._parse_pattern_structure()
    
    def _parse_pattern_structure(self) -> Dict[str, Any]:
        import re
        pattern = self.cypher_pattern
        structure = {
            'node_labels': [],
            'rel_types': [],
            'num_hops': 0,
            'has_properties': False,
            'is_optional': False
        }
        if 'OPTIONAL' in pattern.upper():
            structure['is_optional'] = True
        node_labels = re.findall(r'\((?:[a-z0-9_]*:)?([A-Z][a-zA-Z0-9_]*)', pattern)
        structure['node_labels'] = list(set(node_labels))
        rel_types = re.findall(r'-\[:([A-Za-z_][A-Za-z0-9_]*)\]-', pattern)
        structure['rel_types'] = rel_types
        structure['num_hops'] = len(rel_types)
        if '{' in pattern and '}' in pattern:
            structure['has_properties'] = True
        return structure
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'triple_id': self.triple_id,
            'pattern_id': self.pattern_id,
            'cypher_pattern': self.cypher_pattern,
            'cypher_query': self.cypher_pattern,
            'pattern_signature': self.pattern_signature,
            'semantic_description': self.semantic_description,
            'pattern_structure': self.pattern_structure,
            'anchor_nodes': self.anchor_nodes,
            'hop_count': self.hop_count,
            'sample_paths_total_hits': self.sample_paths_total_hits,
            'score': self.score,
            'is_optional': self.is_optional,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'num_sample_paths': self.num_sample_paths,
            'sample_paths': self.sample_paths,
            'debug_info': self.debug_info,
        }
    
    def get_text_representation(self) -> str:
        return self.semantic_description if self.semantic_description else self.cypher_pattern
    
    def __repr__(self) -> str:
        return f"SubgraphTriple(id={self.triple_id}, nodes={self.num_nodes}, edges={self.num_edges}, score={self.score:.3f})"


def create_triple_from_path_instance(
    path_instance: Any,
    semantic: str = "",
    embedding: Optional[np.ndarray] = None,
    db_name: str = ""
) -> SubgraphTriple:
    """从PathInstance创建SubgraphTriple"""
    
    subgraph = nx.MultiDiGraph()
    
    for i, node in enumerate(path_instance.nodes):
        node_id = str(node.get('id', f'node_{i}'))
        subgraph.add_node(
            node_id,
            label=node.get('label', 'Unknown'),
            **(node.get('properties', {}) or {})
        )
    
    for i, edge in enumerate(path_instance.edges):
        from_id = edge.get('from')
        to_id = edge.get('to')
        if not from_id and i < len(path_instance.nodes):
            from_id = path_instance.nodes[i].get('id', f'node_{i}')
        if not to_id and i + 1 < len(path_instance.nodes):
            to_id = path_instance.nodes[i + 1].get('id', f'node_{i+1}')
        if from_id and to_id:
            edge_attrs = dict(edge.get('properties', {}) or {})
            edge_attrs.setdefault('rel_type', edge.get('rel_type', 'UNKNOWN'))
            subgraph.add_edge(str(from_id), str(to_id), **edge_attrs)
    
    cypher_query = getattr(path_instance, 'cypher_query', '') or f"MATCH {path_instance.get_signature()}"
    if "RETURN" not in cypher_query.upper():
        cypher_query = f"{cypher_query.strip()}\nRETURN *"
    pattern_signature = getattr(path_instance, 'pattern_signature', '') or _extract_match_where_from_query(cypher_query)
    pattern_structure = deepcopy(getattr(path_instance, 'pattern_structure', {})) or _build_pattern_structure_from_path(path_instance)
    sample_paths_attr = getattr(path_instance, 'sample_paths', None)
    if sample_paths_attr:
        sample_paths = [deepcopy(entry) for entry in sample_paths_attr]
    else:
        sample_paths = [{'nodes': deepcopy(path_instance.nodes), 'edges': deepcopy(path_instance.edges)}]
    anchor_nodes = deepcopy(getattr(path_instance, 'anchor_nodes', []))
    sample_paths_total_hits = getattr(path_instance, 'sample_paths_total_hits', len(sample_paths)) or len(sample_paths)
    
    return SubgraphTriple(
        subgraph=subgraph,
        sample_paths=sample_paths,
        cypher_pattern=cypher_query,
        pattern_structure=pattern_structure,
        semantic_description=semantic,
        embedding=embedding,
        pattern_signature=pattern_signature,
        triple_id=path_instance.path_id,
        pattern_id=getattr(path_instance, 'pattern_id', '') or "",
        anchor_nodes=anchor_nodes,
        hop_count=path_instance.get_length(),
        score=path_instance.final_score,
        is_optional=False,
        sample_paths_total_hits=sample_paths_total_hits,
        debug_info={
            'db_name': db_name,
            'pattern_score': getattr(path_instance, 'pattern_score', 0.0),
            'anchor_count': getattr(path_instance, 'anchor_count', 0),
            'semantic_similarity': getattr(path_instance, 'semantic_similarity', 0.0),
        }
    )
def _extract_match_where_from_query(query: str) -> str:
    if not query:
        return ""
    upper_q = query.upper()
    idx = upper_q.find("RETURN")
    if idx != -1:
        return query[:idx].strip()
    return query.strip()
