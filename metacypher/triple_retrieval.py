"""
子图检索系统 - 三元组增强版

版本: 2.0
日期: 2025-12-31
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from subgraph_retrieval import (
    SubgraphRetriever as BaseSubgraphRetriever,
    PathInstance,
    RetrievalConfig
)
from subgraph_triple import SubgraphTriple, create_triple_from_path_instance

logger = logging.getLogger(__name__)


class EnhancedSubgraphRetriever(BaseSubgraphRetriever):
    """增强型子图检索器 - 支持三元组输出"""
    
    def __init__(
        self,
        config: RetrievalConfig,
        schema: Dict[str, Any],
        full_schema_path: str = "",
        enable_semantics: bool = False,
        template_base_path: str = "",
        catalog: Any = None
    ):
        super().__init__(config, schema, full_schema_path, catalog=catalog)
        self.enable_semantics = enable_semantics
        self.template_base_path = template_base_path
        self._current_graph_name: str = ""
        logger.info(f"[EnhancedRetriever] 初始化 - 语义生成: {'启用' if enable_semantics else '禁用'}")
    
    def retrieve_with_triples(
        self,
        pool: Any,
        db_name: str,
        case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行检索并返回三元组格式"""
        
        question = case.get('question') or case.get('nl_question', '')
        logger.info(f"[EnhancedRetriever] 开始检索 - {db_name}: {question}")
        
        self._current_graph_name = db_name
        
        base_result = self.retrieve(pool, db_name, case)
        final_paths = base_result.get('final_paths', [])
        logger.info(f"[EnhancedRetriever] 检索到 {len(final_paths)} 条路径")
        
        triples = self._convert_paths_to_triples(final_paths, db_name)
        logger.info(f"[EnhancedRetriever] 生成 {len(triples)} 个三元组")
        
        return {
            'triples': triples,
            'num_triples': len(triples),
            'pattern_paths': base_result.get('pattern_paths', []),
            'beam_paths': base_result.get('beam_paths', []),
            'final_paths': final_paths,
            'top_k_paths': final_paths,
            'stats': base_result.get('stats', {}),
            'num_patterns_used': base_result.get('stats', {}).get('pattern_paths_count', 0),
            'total_paths_found': base_result.get('stats', {}).get('total_paths_count', 0),
            'strategy_type': 'balanced_expansion_beam_search',
            'db_name': db_name,
            'qid': case.get('qid', ''),
            'question': question,
        }
    
    def _convert_paths_to_triples(
        self,
        path_instances: List[PathInstance],
        db_name: str
    ) -> List[SubgraphTriple]:
        """将PathInstance列表转换为三元组列表"""
        
        if hasattr(self, "_ensure_paths_have_descriptions"):
            self._ensure_paths_have_descriptions(path_instances)
        
        triples = []
        for i, path_instance in enumerate(path_instances, 1):
            try:
                semantic = path_instance.description or path_instance.get_signature()
                triple = create_triple_from_path_instance(
                    path_instance=path_instance,
                    semantic=semantic,
                    embedding=None,
                    db_name=db_name
                )
                triples.append(triple)
                
                if i % 10 == 0 or i == len(path_instances):
                    logger.info(f"  转换进度: {i}/{len(path_instances)}")
            except Exception as e:
                logger.error(f"[ConvertTriples] 路径 {i} 转换失败: {e}")
                continue
        
        logger.info(f"[ConvertTriples] 成功转换 {len(triples)}/{len(path_instances)}")
        return triples
    
    def _generate_semantic(self, path_instance: PathInstance) -> str:
        """生成语义描述（如果启用）"""
        try:
            from pattern_generation import generate_semantic_from_pattern

            cypher_pattern = f"MATCH {path_instance.get_signature()}"

            # Use the real graph name and template directory.
            graph_name = self._current_graph_name or ""
            template_base_path = self.template_base_path or ""

            return generate_semantic_from_pattern(
                graph_name=graph_name,
                pattern=cypher_pattern,
                template_base_path=template_base_path,
                use_cache=True
            )
        except Exception as e:
            logger.warning(f"[GenerateSemantic] 失败: {e}")
            return path_instance.get_signature()
    
    def export_triples_to_json(self, triples: List[SubgraphTriple], output_path: Path) -> None:
        """导出三元组到JSON"""
        export_data = {'num_triples': len(triples), 'triples': []}
        for triple in triples:
            triple_dict = triple.to_dict()
            if 'subgraph' in triple_dict:
                del triple_dict['subgraph']
            if 'embedding' in triple_dict and triple_dict['embedding'] is not None:
                triple_dict['embedding'] = triple_dict['embedding'].tolist()
            export_data['triples'].append(triple_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"[Export] 三元组导出到: {output_path}")


def retrieve_subgraph_triples(
    pool: Any,
    db_name: str,
    case: Dict[str, Any],
    schema: Dict[str, Any],
    config: Optional[RetrievalConfig] = None,
    full_schema_path: str = "",
    enable_semantics: bool = False,
    template_base_path: str = ""
) -> Dict[str, Any]:
    """便捷函数: 执行三元组检索"""
    
    if config is None:
        config = RetrievalConfig(
            use_beam_search=True,
            beam_width=20,
            max_expansion_depth=2,
            top_k_paths=20,
        )
    
    retriever = EnhancedSubgraphRetriever(
        config=config,
        schema=schema,
        full_schema_path=full_schema_path,
        enable_semantics=enable_semantics,
        template_base_path=template_base_path
    )
    return retriever.retrieve_with_triples(pool, db_name, case)
