import os
import json
import torch
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

from transformers import AutoTokenizer, AutoModel


# 配置路径（来自 config，可用环境变量覆盖）
from config import EMBED_MODEL_PATH as MODEL_PATH
from config import FAISS_INDEX_DIR

INDEX_DIR = str(FAISS_INDEX_DIR)


@dataclass
class RetrievalResult:
    """检索结果"""
    pattern_id: str
    cypher_pattern: str
    path_structure: str
    semantic_score: float
    structure_score: float
    hybrid_score: float
    matched_text: str
    semantic_type: str
    matched_nodes: List[str]
    matched_relations: List[str]
    metadata: Dict


class MultiDatabasePathRetriever:
    """多数据库路径模式检索器"""
    
    def __init__(
        self, 
        model_path: str = MODEL_PATH,
        index_dir: str = INDEX_DIR,
        semantic_weight: float = 0.6,
        structure_weight: float = 0.4
    ):
        """
        初始化多数据库检索器
        
        Args:
            model_path: BGE-M3模型路径
            index_dir: 索引根目录
            semantic_weight: 语义相似度权重
            structure_weight: 结构匹配度权重
        """
        self.index_dir = index_dir
        self.semantic_weight = semantic_weight
        self.structure_weight = structure_weight
        
        # 存储所有数据库的索引
        self.databases = {}  # {db_name: {'index': ..., 'metadata': ..., 'stats': ...}}
        
        # 加载BGE-M3模型
        print(f"Loading BGE-M3 model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model loaded on {self.device}")
        
        # 预加载所有数据库索引
        self._load_all_databases()
        
        print(f"\n✓ Multi-database retriever initialized")
        print(f"  Loaded databases: {list(self.databases.keys())}")
        print(f"  Weights: semantic={semantic_weight:.2f}, structure={structure_weight:.2f}")
    
    def _load_all_databases(self):
        """预加载所有数据库的索引"""
        print(f"\nScanning for database indices in {self.index_dir}...")
        
        if not os.path.exists(self.index_dir):
            raise FileNotFoundError(f"Index directory not found: {self.index_dir}")
        
        # 遍历索引目录
        for db_name in os.listdir(self.index_dir):
            db_dir = os.path.join(self.index_dir, db_name)
            
            if not os.path.isdir(db_dir):
                continue
            
            # 检查必需文件
            index_path = os.path.join(db_dir, 'path_hybrid.index')
            metadata_path = os.path.join(db_dir, 'path_hybrid.meta.json')
            
            if not os.path.exists(index_path) or not os.path.exists(metadata_path):
                print(f"  ⚠️  Skipping {db_name}: missing index or metadata")
                continue
            
            try:
                # 加载FAISS索引
                index = faiss.read_index(index_path)
                
                # 加载元数据
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # 加载统计信息(可选)
                stats = None
                stats_path = os.path.join(db_dir, 'index_stats.json')
                if os.path.exists(stats_path):
                    with open(stats_path, 'r', encoding='utf-8') as f:
                        stats = json.load(f)
                
                # 存储
                self.databases[db_name] = {
                    'index': index,
                    'metadata': metadata,
                    'stats': stats
                }
                
                print(f"  ✓ Loaded {db_name}: {index.ntotal} vectors, "
                      f"{stats['num_patterns'] if stats else '?'} patterns")
                
            except Exception as e:
                print(f"  ✗ Failed to load {db_name}: {e}")
                continue
        
        if not self.databases:
            raise RuntimeError("No database indices loaded!")
    
    def get_available_databases(self) -> List[str]:
        """获取可用的数据库列表"""
        return list(self.databases.keys())
    
    @torch.no_grad()
    def encode_query(self, query: str) -> np.ndarray:
        """编码用户查询"""
        prefix = 'Represent this sentence for searching relevant passages: '
        query_text = prefix + query
        
        encoded = self.tokenizer(
            [query_text],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.model(**encoded)
        
        embeddings = self._mean_pooling(
            outputs.last_hidden_state,
            encoded['attention_mask']
        )
        
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().astype('float32')
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling with attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def extract_schema_info(self, analysis: Dict) -> Tuple[Set[str], Set[str]]:
        """从analysis中提取相关的节点和关系标签"""
        node_labels = set()
        relation_types = set()
        
        # 1. 从related_schema中提取
        related_schema = analysis.get('related_schema', {})
        
        # 节点标签
        node_labels.update(related_schema.get('node_labels', []))
        
        # 从node_label_candidates中提取
        for candidate in related_schema.get('node_label_candidates', []):
            if 'label' in candidate:
                node_labels.add(candidate['label'])
        
        # 关系类型
        relation_types.update(related_schema.get('relation_types', []))
        
        # 从relation_candidates中提取
        for candidate in related_schema.get('relation_candidates', []):
            if 'type' in candidate:
                relation_types.add(candidate['type'])
            # 也提取主客体节点标签
            if 'subj_label' in candidate:
                node_labels.add(candidate['subj_label'])
            if 'obj_label' in candidate:
                node_labels.add(candidate['obj_label'])
        
        # 2. 从entities中提取mapped_to的标签
        for entity in analysis.get('entities', []):
            for mapping in entity.get('mapped_to', []):
                if 'label' in mapping:
                    node_labels.add(mapping['label'])
        
        return node_labels, relation_types
    
    def calculate_structure_score(
        self,
        pattern_meta: Dict,
        query_nodes: Set[str],
        query_relations: Set[str]
    ) -> Tuple[float, List[str], List[str]]:
        """计算结构匹配度分数"""
        # 获取路径模式中的节点和关系
        pattern_nodes = set(pattern_meta.get('metadata', {}).get('nodes', []))
        pattern_relations = set(pattern_meta.get('metadata', {}).get('relationships', []))
        
        # 计算匹配
        matched_nodes = list(pattern_nodes & query_nodes)
        matched_relations = list(pattern_relations & query_relations)
        
        # 计算分数
        if not pattern_nodes and not pattern_relations:
            return 0.0, matched_nodes, matched_relations
        
        # 节点匹配度
        if pattern_nodes:
            node_score = len(matched_nodes) / len(pattern_nodes)
        else:
            node_score = 0.0
        
        # 关系匹配度
        if pattern_relations:
            relation_score = len(matched_relations) / len(pattern_relations)
        else:
            relation_score = 0.0
        
        # 综合分数
        if pattern_nodes and pattern_relations:
            structure_score = (node_score + relation_score) / 2
        elif pattern_nodes:
            structure_score = node_score
        elif pattern_relations:
            structure_score = relation_score
        else:
            structure_score = 0.0
        
        return structure_score, matched_nodes, matched_relations
    
    def retrieve_with_hybrid_scoring(
        self,
        query: str,
        graph_name: str,
        analysis: Dict,
        top_k: int = 5,
        retrieval_k: int = 50
    ) -> List[RetrievalResult]:
        """使用混合评分检索路径模式"""
        # 检查数据库是否存在
        if graph_name not in self.databases:
            raise ValueError(
                f"Database '{graph_name}' not found. "
                f"Available databases: {list(self.databases.keys())}"
            )
        
        db = self.databases[graph_name]
        index = db['index']
        metadata = db['metadata']
        
        # 1. 提取结构信息
        query_nodes, query_relations = self.extract_schema_info(analysis)
        
        # 2. 语义检索(获取更多候选)
        query_emb = self.encode_query(query)
        scores, indices = index.search(query_emb, min(retrieval_k, index.ntotal))
        
        # 3. 对每个候选计算混合分数
        hybrid_results = []
        seen_patterns = set()
        
        for semantic_score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            
            meta = metadata[idx]
            pattern_id = meta['pattern_id']
            
            # 去重:每个pattern只保留最高语义分的匹配
            if pattern_id in seen_patterns:
                continue
            seen_patterns.add(pattern_id)
            
            # 计算结构匹配度
            structure_score, matched_nodes, matched_relations = \
                self.calculate_structure_score(meta, query_nodes, query_relations)
            
            # 混合分数
            hybrid_score = (
                self.semantic_weight * float(semantic_score) +
                self.structure_weight * structure_score
            )
            
            result = RetrievalResult(
                pattern_id=pattern_id,
                cypher_pattern=meta['cypher_pattern'],
                path_structure=meta['path_structure'],
                semantic_score=float(semantic_score),
                structure_score=structure_score,
                hybrid_score=hybrid_score,
                matched_text=meta['text'],
                semantic_type=meta['semantic_type'],
                matched_nodes=matched_nodes,
                matched_relations=matched_relations,
                metadata=meta.get('metadata', {})
            )
            
            hybrid_results.append(result)
        
        # 4. 按混合分数排序并返回top-k
        hybrid_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        return hybrid_results[:top_k]
    
    def process_jsonl_file(
        self,
        input_jsonl: str,
        output_jsonl: str,
        top_k: int = 5,
        retrieval_k: int = 50
    ):
        """批量处理JSONL文件"""
        print(f"\n{'='*80}")
        print(f"Processing JSONL file: {input_jsonl}")
        print(f"{'='*80}")
        
        processed = 0
        failed = 0
        graph_stats = defaultdict(int)  # 统计每个graph的查询数量
        
        with open(input_jsonl, 'r', encoding='utf-8') as infile, \
             open(output_jsonl, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                try:
                    # 解析输入
                    data = json.loads(line.strip())
                    
                    # 提取必要信息
                    qid = data.get('qid', f'unknown_{line_num}')
                    graph_name = data.get('graph', '')
                    nl_question = data.get('nl_question', '')
                    analysis = data.get('analysis', {})
                    
                    # 验证必需字段
                    if not graph_name:
                        print(f"  [{line_num}] ⚠️  Skipping {qid}: no 'graph' field")
                        failed += 1
                        continue
                    
                    if not nl_question:
                        print(f"  [{line_num}] ⚠️  Skipping {qid}: no 'nl_question'")
                        failed += 1
                        continue
                    
                    # 检查graph是否存在
                    if graph_name not in self.databases:
                        print(f"  [{line_num}] ⚠️  Skipping {qid}: graph '{graph_name}' not loaded")
                        failed += 1
                        continue
                    
                    # 执行混合检索
                    print(f"  [{line_num}] {graph_name}/{qid}...", end=' ')
                    
                    results = self.retrieve_with_hybrid_scoring(
                        query=nl_question,
                        graph_name=graph_name,
                        analysis=analysis,
                        top_k=top_k,
                        retrieval_k=retrieval_k
                    )
                    
                    # 构建输出(添加retrieved_patterns字段)
                    data['retrieved_patterns'] = {
                        'top_k': top_k,
                        'graph_name': graph_name,
                        'retrieval_method': 'hybrid_semantic_structure',
                        'weights': {
                            'semantic': self.semantic_weight,
                            'structure': self.structure_weight
                        },
                        'patterns': [
                            {
                                'rank': i + 1,
                                'pattern_id': result.pattern_id,
                                'cypher_pattern': result.cypher_pattern,
                                'path_structure': result.path_structure,
                                'scores': {
                                    'semantic': round(result.semantic_score, 4),
                                    'structure': round(result.structure_score, 4),
                                    'hybrid': round(result.hybrid_score, 4)
                                },
                                'matched_via': result.semantic_type,
                                'matched_text': result.matched_text,
                                'structure_match': {
                                    'matched_nodes': result.matched_nodes,
                                    'matched_relations': result.matched_relations
                                },
                                'metadata': result.metadata
                            }
                            for i, result in enumerate(results)
                        ]
                    }
                    
                    # 写入输出
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    processed += 1
                    graph_stats[graph_name] += 1
                    
                    print(f"✓ ({len(results)} patterns)")
                    
                except Exception as e:
                    print(f"  [{line_num}] ✗ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    failed += 1
                    continue
        
        # 打印统计信息
        print(f"\n{'='*80}")
        print(f"Processing completed!")
        print(f"{'='*80}")
        print(f"Total processed: {processed}")
        print(f"Total failed:    {failed}")
        print(f"\nQueries per database:")
        for graph_name in sorted(graph_stats.keys()):
            print(f"  {graph_name}: {graph_stats[graph_name]}")
        print(f"\nOutput saved to: {output_jsonl}")
        print(f"{'='*80}\n")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Multi-Database Hybrid Path Pattern Retriever',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a JSONL file (graph names from each line's 'graph' field)
  python path_search_multi_db.py input.jsonl -o output.jsonl
  
  # Adjust scoring weights
  python path_search_multi_db.py input.jsonl --semantic-weight 0.7 --structure-weight 0.3
  
  # Get more candidates for reranking
  python path_search_multi_db.py input.jsonl --retrieval-k 100

Note:
  - Each line in the JSONL must have a 'graph' field specifying the database name
  - All available database indices are preloaded at startup
  - Indices must be in {INDEX_DIR}/{{graph_name}}/path_hybrid.index
        """
    )
    
    parser.add_argument('input_jsonl', help='Input JSONL file')
    parser.add_argument('-o', '--output', help='Output JSONL file (default: input_enhanced.jsonl)')
    parser.add_argument('--top-k', type=int, default=10, help='Number of patterns to retrieve (default: 10)')
    parser.add_argument('--retrieval-k', type=int, default=50, 
                       help='Number of candidates for reranking (default: 50)')
    parser.add_argument('--semantic-weight', type=float, default=0.7,
                       help='Semantic similarity weight (default: 0.7)')
    parser.add_argument('--structure-weight', type=float, default=0.3,
                       help='Structure matching weight (default: 0.3)')

    args = parser.parse_args()
    
    # 设置输出文件
    if not args.output:
        input_path = Path(args.input_jsonl)
        args.output = str(input_path.parent / f"{input_path.stem}_enhanced.jsonl")
    
    # 验证权重
    if abs(args.semantic_weight + args.structure_weight - 1.0) > 0.01:
        print(f"⚠️  Warning: Weights don't sum to 1.0 "
              f"({args.semantic_weight} + {args.structure_weight} = "
              f"{args.semantic_weight + args.structure_weight})")
    
    # 初始化检索器(预加载所有数据库)
    retriever = MultiDatabasePathRetriever(
        semantic_weight=args.semantic_weight,
        structure_weight=args.structure_weight
    )
    
    # 批量处理
    retriever.process_jsonl_file(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output,
        top_k=args.top_k,
        retrieval_k=args.retrieval_k
    )
    
    print("✓ Done!")


if __name__ == "__main__":
    main()