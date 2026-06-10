"""
Beam-search pipeline components for subgraph retrieval.

Extracted from subgraph_retrieval.py:
  - BGEEmbedder          — BGE-M3 encoder wrapper
  - RepetitionDetector   — detects repeated-relation requirements
  - BalancedCandidateGenerator — open-ended expansion candidate generation
  - SemanticFilter       — semantic + prior-score ranking
  - OptionalMatchExecutor — executes OPTIONAL MATCH expansions
  - SemanticGuidedBeamSearch — top-level beam-search orchestrator
"""

import logging
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from config import ABLATE_ADAPTIVE_EXPANSION
from retrieval_config import RetrievalConfig
from path_model import (
    PathInstance,
    RepetitionInfo,
    ExpansionCandidate,
    compute_related_schema_coverage,
    build_pattern_structure,
    extract_anchor_nodes_from_path,
)
from cypher_utils import build_match_clause_and_var_map
from schema_template import PathDescriptionBuilder

logger = logging.getLogger(__name__)


# ============================================================================
# BGE-M3 Embedding接口
# ============================================================================

class BGEEmbedder:
    """BGE-M3 Embedding模型封装"""

    def __init__(self, model_name: str = 'BAAI/bge-m3', device: str = 'cuda'):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """加载模型"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            logger.info(f"[BGE] 加载模型 {self.model_name}...")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)

            if self.device == 'cuda' and torch.cuda.is_available():
                self.model = self.model.cuda()
            else:
                self.device = 'cpu'
                self.model = self.model.cpu()

            self.model.eval()
            logger.info(f"[BGE] 模型加载成功 (device={self.device})")

        except Exception as e:
            logger.warning(f"[BGE] 无法加载transformers模型: {e}")
            logger.warning(f"[BGE] 使用Mock模式（随机向量）")
            self.model = None

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """批量编码文本"""
        if self.model is None or self.tokenizer is None:
            return np.random.randn(len(texts), 768).astype(np.float32)

        try:
            import torch

            if self.model is None or self.tokenizer is None:
                return np.random.randn(len(texts), 768).astype(np.float32)

            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )

                if self.device == 'cuda':
                    encoded = {k: v.cuda() for k, v in encoded.items()}

                with torch.no_grad():
                    outputs = self.model(**encoded)
                    embeddings = outputs.last_hidden_state.mean(dim=1)

                all_embeddings.append(embeddings.cpu().numpy())

            return np.vstack(all_embeddings)

        except Exception as e:
            logger.error(f"[BGE] 编码失败: {e}")
            return np.random.randn(len(texts), 768).astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """编码单个文本"""
        return self.encode([text])[0]


# ============================================================================
# 重复关系检测器
# ============================================================================

class RepetitionDetector:
    """检测问题中的重复关系需求"""

    REPETITION_TRIGGERS = {
        'at least two': 2,
        'at least 2': 2,
        'at least three': 3,
        'at least 3': 3,
        'both': 2,
        'two': 2,
        'three': 3,
        'multiple': 2
    }

    def __init__(self, question: str, question_analysis: Optional[Dict] = None):
        self.question = question.lower()
        self.question_analysis = question_analysis or {}

    def detect(self) -> List[RepetitionInfo]:
        """检测重复关系需求"""
        repetitions = []
        repetitions.extend(self._detect_from_text())
        if self.question_analysis:
            repetitions.extend(self._detect_from_analysis())

        if repetitions:
            logger.info(f"[RepetitionDetector] 检测到 {len(repetitions)} 个重复关系需求")
            for rep in repetitions:
                logger.info(f"  • {rep.relation_type} (至少{rep.min_count}次)")

        return repetitions

    def _detect_from_text(self) -> List[RepetitionInfo]:
        """从问题文本检测"""
        repetitions = []

        for trigger, count in self.REPETITION_TRIGGERS.items():
            if trigger in self.question:
                if 'flow' in self.question:
                    relation_type = 'flowsThrough' if 'through' in self.question else 'flowsInto'
                elif 'located' in self.question or ' in ' in self.question:
                    relation_type = 'locatedIn'
                elif 'part of' in self.question:
                    relation_type = 'partOf'
                else:
                    continue

                repetitions.append(RepetitionInfo(
                    relation_type=relation_type,
                    min_count=count,
                    constraint_type='different_instances'
                ))
                logger.debug(f"  触发词 '{trigger}' → {relation_type} ×{count}")

        return repetitions

    def _detect_from_analysis(self) -> List[RepetitionInfo]:
        """从问题分析检测"""
        repetitions = []
        constraints = self.question_analysis.get('constraints', [])

        for constraint in constraints:
            if constraint.get('type') == 'cardinality':
                repetitions.append(RepetitionInfo(
                    relation_type=constraint.get('relation'),
                    min_count=constraint.get('min_count', 2),
                    constraint_type='different_instances'
                ))

        return repetitions


# ============================================================================
# 平衡候选生成器（核心）
# ============================================================================

class BalancedCandidateGenerator:

    def __init__(
        self,
        config: RetrievalConfig,
        schema: Dict,
        question: str,
        question_analysis: Optional[Dict] = None,
        relation_index: Optional[Dict] = None,
        description_builder: Optional[PathDescriptionBuilder] = None
    ):
        self.config = config
        self.schema = schema
        self.question = question
        self.question_analysis = question_analysis or {}
        self.relation_index = relation_index
        self.description_builder = description_builder

        if relation_index:
            self.relation_map = self._convert_index_to_map(relation_index)
            logger.info(f"[BalancedCandGen] 使用提供的relation_index")
        else:
            self._build_schema_index()
            logger.info(f"[BalancedCandGen] 使用简化schema构建索引")

        self.related_nodes = self._extract_related_nodes()

        if self.config.enable_repetition_detection:
            self.repetition_detector = RepetitionDetector(question, question_analysis)
            self.repetitions = self.repetition_detector.detect()
        else:
            self.repetitions = []

        logger.info(f"[BalancedCandGen] 初始化完成")
        logger.info(f"  • Related nodes: {self.related_nodes}")
        logger.info(f"  • Repetitions: {len(self.repetitions)}")

    def _convert_index_to_map(self, relation_index: Dict) -> Dict:
        relation_map: Dict[str, List] = defaultdict(list)

        for (node_label, direction_key), relations in relation_index.items():
            for rel_type, target_label in relations:
                if direction_key == 'out':
                    relation_map[node_label].append({
                        'type': rel_type,
                        'direction': 'forward',
                        'target': target_label
                    })
                elif direction_key == 'in':
                    relation_map[node_label].append({
                        'type': rel_type,
                        'direction': 'backward',
                        'target': target_label
                    })

        return dict(relation_map)

    def _build_schema_index(self):
        """构建schema索引"""
        self.relation_map: Dict[str, List] = defaultdict(list)

        relationships = self.schema.get('relationships', [])
        for rel in relationships:
            from_label = rel.get('from')
            to_label = rel.get('to')
            rel_type = rel.get('type')

            self.relation_map[from_label].append({
                'type': rel_type,
                'direction': 'forward',
                'target': to_label
            })
            self.relation_map[to_label].append({
                'type': rel_type,
                'direction': 'backward',
                'target': from_label
            })

    def _extract_related_nodes(self) -> Set[str]:
        """从问题分析提取相关节点类别"""
        related: Set[str] = set()

        if not self.question_analysis:
            return related

        entities = self.question_analysis.get('entities', [])
        for entity in entities:
            for mapping in entity.get('mapped_to', []):
                label = mapping.get('label', '')
                if label:
                    related.add(label)
            for cand in entity.get('faiss_candidates', []):
                label = cand.get('node_label', '')
                if label:
                    related.add(label)

        related_schema = self.question_analysis.get('related_schema', {})
        for label in related_schema.get('node_labels', []):
            related.add(label)

        return related

    def generate_candidates(self, paths: List[PathInstance]) -> List[ExpansionCandidate]:
        """生成候选（平衡方案）"""
        all_candidates: List[ExpansionCandidate] = []

        for path in paths:
            if self.config.expand_from_all_nodes:
                expansion_nodes = list(range(len(path.nodes)))
            else:
                expansion_nodes = [0, len(path.nodes) - 1] if len(path.nodes) > 1 else [0]

            for node_idx in expansion_nodes:
                node = path.nodes[node_idx]
                node_label = node.get('label')
                node_id = node.get('id')

                if not node_label or not node_id:
                    continue

                assert node_label is not None
                assert node_id is not None

                if node_idx == 0 and 'backward' in self.config.expansion_directions:
                    all_candidates.extend(
                        self._generate_from_node(path, node_idx, node_label, node_id, 'backward')
                    )

                if node_idx == len(path.nodes) - 1 and 'forward' in self.config.expansion_directions:
                    all_candidates.extend(
                        self._generate_from_node(path, node_idx, node_label, node_id, 'forward')
                    )

                if 0 < node_idx < len(path.nodes) - 1:
                    if 'forward' in self.config.expansion_directions:
                        all_candidates.extend(
                            self._generate_from_node(path, node_idx, node_label, node_id, 'forward')
                        )
                    if 'backward' in self.config.expansion_directions:
                        all_candidates.extend(
                            self._generate_from_node(path, node_idx, node_label, node_id, 'backward')
                        )

        logger.info(f"[BalancedCandGen] 生成 {len(all_candidates)} 个候选（包含重复）")
        return all_candidates

    def _generate_from_node(
        self,
        path: PathInstance,
        node_idx: int,
        node_label: str,
        node_id: str,
        direction: str
    ) -> List[ExpansionCandidate]:
        candidates: List[ExpansionCandidate] = []

        possible_relations = self.relation_map.get(node_label, [])
        valid_relations = [r for r in possible_relations if r['direction'] == direction]
        valid_relations = valid_relations[:self.config.max_relations_per_node]

        for rel_info in valid_relations:
            rel_type = rel_info['type']
            target_label = rel_info['target']
            repeat_count = self._get_repeat_count(rel_type)

            if repeat_count > 1:
                for instance_id in range(repeat_count):
                    candidate = self._create_candidate(
                        path, node_idx, node_label, node_id,
                        rel_type, target_label, direction,
                        is_repeated=True, instance_id=instance_id
                    )
                    candidates.append(candidate)
            else:
                candidate = self._create_candidate(
                    path, node_idx, node_label, node_id,
                    rel_type, target_label, direction,
                    is_repeated=False, instance_id=0
                )
                candidates.append(candidate)

        return candidates

    def _get_repeat_count(self, rel_type: str) -> int:
        """获取关系需要重复的次数"""
        if ABLATE_ADAPTIVE_EXPANSION:
            return 1
        for rep in self.repetitions:
            if rep.relation_type == rel_type:
                return rep.min_count
        return 1

    def _create_candidate(
        self,
        path: PathInstance,
        node_idx: int,
        node_label: str,
        node_id: str,
        rel_type: str,
        target_label: str,
        direction: str,
        is_repeated: bool,
        instance_id: int
    ) -> ExpansionCandidate:
        description = self._generate_description(
            path=path, node_idx=node_idx, from_label=node_label,
            rel_type=rel_type, to_label=target_label, direction=direction,
            is_repeated=is_repeated, instance_id=instance_id
        )

        # Ablation (tab:ablation_overall, "w/o adaptive expansion"): expansion
        # stays schema-valid but question-blind — the question-derived prior
        # and related-schema coverage scores are neutralized to a constant.
        if ABLATE_ADAPTIVE_EXPANSION:
            prior_score = 0.0
            coverage_score = 0.0
        else:
            prior_score = self._calculate_prior_score(target_label, is_repeated, path)
            related_schema = (self.question_analysis or {}).get('related_schema', {})
            expected_nodes = {n.get('label') for n in path.nodes if n.get('label')}
            expected_rels = {edge.get('rel_type') for edge in path.edges if edge.get('rel_type')}
            if target_label:
                expected_nodes.add(target_label)
            if rel_type:
                expected_rels.add(rel_type)
            coverage_score = compute_related_schema_coverage(
                expected_nodes, expected_rels, related_schema,
                beta=self.config.coverage_beta,
                use_length_gain=self.config.coverage_use_length_gain,
                length_tau=self.config.coverage_length_tau
            )

        constraint = None
        if is_repeated and instance_id > 0:
            constraint = self._generate_repetition_constraint(rel_type, target_label, instance_id)

        return ExpansionCandidate(
            base_path_id=path.path_id,
            direction=direction,
            expand_from_node_idx=node_idx,
            expand_from_label=node_label,
            expand_from_id=node_id,
            relation_type=rel_type,
            target_label=target_label,
            is_repeated=is_repeated,
            instance_id=instance_id,
            repetition_constraint=constraint,
            prior_score=prior_score,
            pattern_description=description,
            coverage_score=coverage_score
        )

    def _calculate_prior_score(
        self,
        target_label: str,
        is_repeated: bool,
        path: PathInstance
    ) -> float:
        score = 0.0

        if target_label in self.related_nodes:
            score += self.config.related_node_bonus
            logger.debug(f"  • {target_label}: related_node +{self.config.related_node_bonus}")

        path_labels = {n.get('label') for n in path.nodes}
        if target_label in path_labels:
            score += self.config.in_path_bonus
            logger.debug(f"  • {target_label}: in_path +{self.config.in_path_bonus}")

        if is_repeated:
            score += self.config.repeated_relation_bonus
            logger.debug(f"  • repeated_relation +{self.config.repeated_relation_bonus}")

        return score

    def _generate_description(
        self,
        path: PathInstance,
        node_idx: int,
        from_label: str,
        rel_type: str,
        to_label: str,
        direction: str,
        is_repeated: bool,
        instance_id: int
    ) -> str:
        if self.description_builder:
            base_desc = self.description_builder.build(path, use_optional=False, include_constraints=True)
        else:
            base_desc = self._path_to_text(path)

        expansion_sentence = self._build_expansion_sentence(
            path=path, node_idx=node_idx, rel_type=rel_type,
            target_label=to_label, direction=direction,
            is_repeated=is_repeated, instance_id=instance_id
        )

        if base_desc and expansion_sentence:
            return f"{base_desc} {expansion_sentence}".strip()
        return base_desc or expansion_sentence

    def _path_to_text(self, path: PathInstance) -> str:
        parts = []
        for i, node in enumerate(path.nodes):
            label = node.get('label', 'entity')
            parts.append(label)
            if i < len(path.edges):
                edge = path.edges[i]
                rel_type = edge.get('rel_type', 'relates to')
                rel_text = self._relation_to_text(rel_type)
                parts.append(rel_text)
        return ' '.join(parts)

    def _relation_to_text(self, rel_type: str) -> str:
        mapping = {
            'flowsThrough': 'flows through',
            'flowsInto': 'flows into',
            'locatedIn': 'is located in',
            'partOf': 'is part of',
            'hasCapital': 'has capital',
            'originatesFrom': 'originates from',
        }
        return mapping.get(rel_type, rel_type.lower().replace('_', ' '))

    def _get_node_display_name(self, node: Dict[str, Any]) -> str:
        props = node.get('properties') or {}
        if 'name' in props:
            return props['name']
        if node.get('display_name'):
            return node['display_name']
        label = node.get('label')
        if label:
            return f"unspecified {label.lower()}"
        return "unspecified entity"

    def _build_expansion_sentence(
        self,
        path: PathInstance,
        node_idx: int,
        rel_type: str,
        target_label: str,
        direction: str,
        is_repeated: bool,
        instance_id: int
    ) -> str:
        subj_node = path.nodes[node_idx]
        target_name = self._get_node_display_name({'label': target_label or 'Entity'})
        existing_name = self._get_node_display_name(subj_node)

        sentence = ""
        if self.description_builder:
            temp_nodes, temp_edge = self._build_temp_expansion_path(
                subj_node=subj_node, target_label=target_label,
                rel_type=rel_type, direction=direction
            )
            temp_path = PathInstance(path_id="exp_temp", nodes=temp_nodes, edges=[temp_edge])
            sentence = self.description_builder.build(
                temp_path, use_optional=True, include_constraints=False
            )

        if not sentence:
            if direction == 'backward':
                sentence = f"{target_name} -[{rel_type}]-> {existing_name}"
            else:
                sentence = f"{existing_name} -[{rel_type}]-> {target_name}"

        if is_repeated:
            ordinals = ['first', 'second', 'third', 'fourth', 'fifth']
            ordinal = ordinals[instance_id] if instance_id < len(ordinals) else f"{instance_id+1}th"
            sentence = f"{sentence} (instance: {ordinal})"

        return sentence

    def _build_temp_expansion_path(
        self,
        subj_node: Dict[str, Any],
        target_label: str,
        rel_type: str,
        direction: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        subj_clone = self._clone_node_for_description(
            subj_node, fallback_id=f"temp_subj_{id(subj_node)}"
        )
        new_node_id = f"temp_{target_label or 'Node'}_{subj_clone['id']}"
        new_node = {
            'label': target_label,
            'id': new_node_id,
            'properties': {},
            'display_name': None
        }

        if direction == 'backward':
            nodes = [new_node, subj_clone]
            edge = {
                'rel_type': rel_type,
                'from': new_node_id,
                'to': subj_clone['id'],
                'properties': {}
            }
        else:
            nodes = [subj_clone, new_node]
            edge = {
                'rel_type': rel_type,
                'from': subj_clone['id'],
                'to': new_node_id,
                'properties': {}
            }
        return nodes, edge

    def _clone_node_for_description(self, node: Dict[str, Any], fallback_id: str) -> Dict[str, Any]:
        return {
            'label': node.get('label'),
            'id': node.get('id') or node.get('var') or fallback_id,
            'properties': dict(node.get('properties') or {}),
            'display_name': node.get('display_name')
        }

    def _generate_repetition_constraint(
        self,
        rel_type: str,
        target_label: str,
        instance_id: int
    ) -> str:
        """重复约束不在Cypher层实现，返回空串"""
        return ""


# ============================================================================
# 语义过滤器（集成prior_score）
# ============================================================================

class SemanticFilter:
    """语义过滤器（集成prior_score）"""

    def __init__(
        self,
        config: RetrievalConfig,
        embedder: BGEEmbedder,
        question: str,
        related_schema: Optional[Dict] = None
    ):
        self.config = config
        self.embedder = embedder
        self.question = question
        self.related_schema = related_schema or {}
        self.question_embedding = embedder.encode_single(question)
        logger.info(f"[SemanticFilter] 初始化完成")

    def filter_and_rank(
        self,
        candidates: List[ExpansionCandidate],
        top_k: Optional[int] = None
    ) -> List[ExpansionCandidate]:
        """语义过滤和排序: final_score = semantic + coverage + prior"""
        if not self.config.use_semantic_filtering:
            return candidates[:top_k] if top_k else candidates

        if not candidates:
            return []

        similarities = [0.0 for _ in candidates]
        valid_descriptions = []
        valid_indices = []
        for idx, candidate in enumerate(candidates):
            desc = (candidate.pattern_description or "").strip()
            if desc:
                valid_descriptions.append(desc)
                valid_indices.append(idx)

        if valid_descriptions:
            desc_embeddings = self.embedder.encode(valid_descriptions, self.config.embedding_batch_size)
            valid_similarities = self._cosine_similarity(self.question_embedding, desc_embeddings)
            for sim, idx in zip(valid_similarities, valid_indices):
                similarities[idx] = float(sim)

        weights = self.config.beam_score_weights
        for i, candidate in enumerate(candidates):
            candidate.semantic_similarity = float(similarities[i])
            candidate.final_score = (
                weights.get("semantic", 0.0) * candidate.semantic_similarity +
                weights.get("coverage", 0.0) * candidate.coverage_score +
                weights.get("prior", 0.0) * candidate.prior_score
            )

        filtered = [
            c for c in candidates
            if c.final_score >= self.config.min_beam_final_score
        ]
        logger.info(f"[SemanticFilter] 过滤: {len(candidates)} → {len(filtered)}")

        filtered.sort(key=lambda c: c.final_score, reverse=True)

        if top_k:
            filtered = filtered[:top_k]

        if filtered and logger.isEnabledFor(logging.INFO):
            logger.info(f"[SemanticFilter] Top-3候选:")
            for i, c in enumerate(filtered[:3], 1):
                logger.info(
                    f"  {i}. {c.expand_from_label} -[{c.relation_type}]-> {c.target_label} "
                    f"(sem={c.semantic_similarity:.3f}, prior={c.prior_score:.3f}, "
                    f"cov={c.coverage_score:.3f}, final={c.final_score:.3f})"
                )

        return filtered

    def _cosine_similarity(self, query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        doc_norms = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)
        return np.dot(doc_norms, query_norm)


# ============================================================================
# OPTIONAL MATCH执行器
# ============================================================================

class OptionalMatchExecutor:
    """OPTIONAL MATCH执行器"""

    def __init__(
        self,
        config: RetrievalConfig,
        embedder: Optional[BGEEmbedder] = None,
        question_embedding: Optional[np.ndarray] = None,
        description_builder: Optional[PathDescriptionBuilder] = None,
        related_schema: Optional[Dict] = None
    ):
        self.config = config
        self.embedder = embedder
        self.question_embedding = question_embedding
        self.description_builder = description_builder
        self.related_schema = related_schema or {}
        if question_embedding is not None:
            norm = np.linalg.norm(question_embedding) + 1e-8
            self._normalized_question_embedding = question_embedding / norm
        else:
            self._normalized_question_embedding = None

    def execute_candidates(
        self,
        candidates: List[ExpansionCandidate],
        base_paths: Dict[str, PathInstance],
        pool: Any,
        db_name: str,
        semantic_filter: Optional[SemanticFilter] = None
    ) -> List[PathInstance]:
        """执行候选扩展，使用OPTIONAL MATCH"""
        expanded_paths: List[PathInstance] = []
        logger.info(f"[OptionalMatch] 执行 {len(candidates)} 个候选")

        if semantic_filter:
            candidates = semantic_filter.filter_and_rank(candidates, top_k=self.config.candidate_width)
        else:
            candidates = candidates[:self.config.candidate_width]

        if self.config.debug_print_cypher and candidates:
            logger.info(f"\n{'='*80}")
            logger.info(f"[DEBUG] 扩展路径的Cypher查询（前{min(3, len(candidates))}个）")
            logger.info(f"{'='*80}")

        for idx, candidate in enumerate(candidates):
            base_path = base_paths.get(candidate.base_path_id)
            if not base_path:
                continue

            try:
                query = self._build_optional_match_query(candidate, base_path)

                if self.config.debug_print_cypher and idx < 3:
                    logger.info(f"\n  [{idx+1}] 扩展查询:")
                    logger.info(f"  {'─'*60}")
                    for line in query.split('\n'):
                        logger.info(f"  {line}")
                    logger.info(f"  {'─'*60}")
                    logger.info(f"  Base: {base_path.pattern_id}, 扩展: {candidate.relation_type}\n")

                results = self._execute_query(query, pool, db_name)
                new_paths = self._parse_results(results, candidate, base_path, query)
                expanded_paths.extend(new_paths)

            except Exception as e:
                logger.warning(f"[OptionalMatch] 执行失败: {e}")
                continue

        logger.info(f"[OptionalMatch] 生成 {len(expanded_paths)} 条新路径")
        return expanded_paths

    def _build_optional_match_query(
        self,
        candidate: ExpansionCandidate,
        base_path: PathInstance
    ) -> str:
        from cypher_utils import (
            _format_cypher_date, _format_cypher_value, _parse_numeric_value
        )

        base_match, node_var_map = self._build_base_match(base_path)
        where_clause = self._build_where_clause(base_path, node_var_map)
        optional_match = self._build_optional_match(candidate, base_path, node_var_map)
        return_clause = self._build_return_clause(candidate, base_path, node_var_map)

        parts = [base_match]
        if where_clause:
            parts.append(where_clause)
        parts.append(optional_match)
        parts.append(return_clause)

        query = "\n".join(parts)

        if self.config.debug_print_query:
            logger.debug(f"[Query]\n{query}")

        return query

    def _build_base_match(self, path: PathInstance) -> Tuple[str, Dict[str, str]]:
        return build_match_clause_and_var_map(path)

    def _build_where_clause(self, path: PathInstance, node_var_map: Dict[str, str]) -> str:
        from cypher_utils import (
            _format_cypher_date, _format_cypher_value, _parse_numeric_value
        )

        if not path.constraints:
            return ""

        operator_map = {
            'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>=', 'eq': '=', 'ne': '!='
        }
        conditions = []

        for c in path.constraints:
            node_var = self._resolve_node_var(c, node_var_map)
            prop = c.get('property')
            op = c.get('operator', 'eq')
            value = c.get('value')
            value_type = c.get('value_type', 'int')

            if value_type == 'date':
                cypher_value = _format_cypher_date(value)
                if cypher_value is None:
                    continue
                op_symbol = operator_map.get(op, '=')
                if c.get('allow_null'):
                    conditions.append(
                        f"({node_var}.{prop} {op_symbol} {cypher_value} "
                        f"OR {node_var}.{prop} IS NULL)"
                    )
                else:
                    conditions.append(f"{node_var}.{prop} {op_symbol} {cypher_value}")

            elif value_type == 'list(str)':
                cypher_val = _format_cypher_value(value)
                conditions.append(f"{cypher_val} IN {node_var}.{prop}")

            elif value_type in ['int', 'float']:
                op_symbol = operator_map.get(op, '=')
                parsed_value = _parse_numeric_value(value, allow_float=(value_type == 'float'))
                if parsed_value is None:
                    continue
                conditions.append(f"{node_var}.{prop} {op_symbol} {parsed_value}")

            elif value_type == 'str':
                op_symbol = operator_map.get(op, '=')
                conditions.append(f"{node_var}.{prop} {op_symbol} {_format_cypher_value(value)}")

        if conditions:
            return "WHERE " + " AND ".join(conditions)
        return ""

    def _resolve_node_var(self, constraint: Dict, node_var_map: Dict[str, str]) -> str:
        node_var = constraint.get('node_var')
        if node_var:
            return node_var
        node_id = constraint.get('node_id')
        if node_id and node_var_map.get(node_id):
            return node_var_map[node_id]
        node_idx = constraint.get('node_idx')
        if node_idx is not None:
            return f"n{node_idx}"
        return "n0"

    def _get_candidate_node_var(
        self, candidate: ExpansionCandidate, node_var_map: Dict[str, str]
    ) -> str:
        node_id = candidate.expand_from_id
        if node_id and node_var_map.get(node_id):
            return node_var_map[node_id]
        return f"n{candidate.expand_from_node_idx}"

    def _get_new_node_var_name(self, candidate: ExpansionCandidate) -> str:
        return f"n_new{candidate.instance_id}" if candidate.is_repeated else "n_new"

    def _get_new_rel_var_name(self, candidate: ExpansionCandidate) -> str:
        return f"r_new{candidate.instance_id}" if candidate.is_repeated else "r_new"

    def _build_optional_match(
        self,
        candidate: ExpansionCandidate,
        base_path: PathInstance,
        node_var_map: Dict[str, str]
    ) -> str:
        from_var = self._get_candidate_node_var(candidate, node_var_map)
        new_var = self._get_new_node_var_name(candidate)
        new_rel_var = self._get_new_rel_var_name(candidate)

        rel_type = candidate.relation_type
        target_label = candidate.target_label

        if candidate.direction == 'forward':
            optional = f"OPTIONAL MATCH ({from_var})-[{new_rel_var}:{rel_type}]->({new_var}:{target_label})"
        else:
            optional = f"OPTIONAL MATCH ({from_var})<-[{new_rel_var}:{rel_type}]-({new_var}:{target_label})"

        if candidate.repetition_constraint:
            optional += f"\nWHERE {candidate.repetition_constraint}"

        return optional

    def _build_return_clause(
        self,
        candidate: ExpansionCandidate,
        base_path: PathInstance,
        node_var_map: Dict[str, str]
    ) -> str:
        base_var = self._get_candidate_node_var(candidate, node_var_map)
        new_var = self._get_new_node_var_name(candidate)
        new_rel_var = self._get_new_rel_var_name(candidate)

        return (
            f"RETURN {base_var} AS base_node, "
            f"{new_var} AS new_node, "
            f"{new_rel_var} AS new_rel "
            f"LIMIT {self.config.max_results_per_query}"
        )

    def _build_sample_signature(self, nodes: List[Dict[str, Any]]) -> Tuple[Any, ...]:
        signature = []
        for node in nodes:
            identifier = node.get('id') or node.get('properties', {}).get('name') or node.get('label')
            signature.append(identifier)
        return tuple(signature)

    def _execute_query(self, query: str, pool: Any, db_name: str) -> List[Dict]:
        try:
            if hasattr(pool, 'execute'):
                result = pool.execute(db_name, query)
                return [dict(record) for record in result]
            else:
                with pool.get_connection(db_name) as conn:
                    with conn.session() as session:
                        result = session.run(query)
                        return [dict(record) for record in result]
        except Exception as e:
            logger.warning(f"[OptionalMatch] 查询执行失败: {e}")
            if self.config.debug_print_query:
                logger.debug(f"[FailedQuery]\n{query}")
            return []

    def _parse_results(
        self,
        results: List[Dict],
        candidate: ExpansionCandidate,
        base_path: PathInstance,
        executed_query: str
    ) -> List[PathInstance]:
        new_paths: List[PathInstance] = []
        sample_entries: List[Dict[str, Any]] = []
        seen_signatures: Set[Tuple[Any, ...]] = set()
        max_samples = getattr(self.config, "max_sample_paths_per_triple", 5)

        for record in results:
            new_node_data = record.get('new_node')
            if new_node_data is None:
                new_node_key = self._get_new_node_var_name(candidate)
                new_node_data = record.get(new_node_key)

            if new_node_data is None:
                continue

            base_node_data = record.get('base_node')
            rel_data = record.get('new_rel')

            new_path = deepcopy(base_path)

            new_node_id = new_node_data.get('id')
            if new_node_id is None:
                new_node_id = new_node_data.get('eid')
            if new_node_id is None:
                assert candidate.target_label and candidate.instance_id is not None
                new_node_id = f"{candidate.target_label.lower()}_{len(new_path.nodes)}_{candidate.instance_id}"

            new_node = {
                'label': candidate.target_label,
                'id': new_node_id,
                'var': new_node_id,
                'properties': {},
                'display_name': new_node_data.get('name')
            }

            source_node = new_path.nodes[candidate.expand_from_node_idx]
            source_id = source_node.get('id')

            rel_properties = {}
            if rel_data:
                try:
                    rel_properties = dict(rel_data)
                except Exception:
                    rel_properties = {}

            new_edge = {
                'rel_type': candidate.relation_type,
                'direction': candidate.direction,
                'from': source_id if candidate.direction == 'forward' else new_node_id,
                'to': new_node_id if candidate.direction == 'forward' else source_id,
                'properties': rel_properties
            }

            if candidate.direction == 'forward':
                new_path.nodes.append(new_node)
                new_path.edges.append(new_edge)
            else:
                new_path.nodes.insert(0, new_node)
                new_path.edges.insert(0, new_edge)

            new_path.path_id = f"{base_path.path_id}_exp{len(new_paths)}"

            new_path.expansion_info = {
                'latest_triplet': {
                    'from_node': self._extract_node_snapshot(
                        base_node_data,
                        fallback_label=source_node.get('label'),
                        fallback_id=source_node.get('id')
                    ),
                    'relationship': self._extract_relationship_snapshot(
                        rel_data, candidate.relation_type
                    ),
                    'to_node': self._extract_node_snapshot(
                        new_node_data,
                        fallback_label=candidate.target_label,
                        fallback_id=new_node_id
                    )
                }
            }

            new_path.prior_score = candidate.prior_score
            new_path.cypher_query = executed_query

            self._populate_path_metadata(new_path)

            new_paths.append(new_path)

            sig = self._build_sample_signature(new_path.nodes)
            if sig not in seen_signatures and len(sample_entries) < max_samples:
                sample_entries.append({
                    'nodes': deepcopy(new_path.nodes),
                    'edges': deepcopy(new_path.edges)
                })
                seen_signatures.add(sig)

        total_hits = len(results)
        if not sample_entries and new_paths:
            sample_entries = [{'nodes': deepcopy(new_paths[0].nodes), 'edges': deepcopy(new_paths[0].edges)}]
        for path in new_paths:
            path.sample_paths_total_hits = total_hits
            path.sample_paths = [deepcopy(entry) for entry in sample_entries]

        return new_paths

    def _populate_path_metadata(self, path: PathInstance) -> None:
        if self.description_builder:
            path.description = self.description_builder.build(
                path, use_optional=False, include_constraints=True
            )
        else:
            path.description = ""

        node_labels = {node.get('label') for node in path.nodes if node.get('label')}
        rel_types = {edge.get('rel_type') for edge in path.edges if edge.get('rel_type')}
        path.coverage_score = compute_related_schema_coverage(
            node_labels, rel_types, self.related_schema,
            beta=self.config.coverage_beta,
            use_length_gain=self.config.coverage_use_length_gain,
            length_tau=self.config.coverage_length_tau
        )

        path.semantic_similarity = self._compute_semantic_similarity(path.description)

        weights = self.config.beam_score_weights
        path.final_score = (
            weights.get("semantic", 0.0) * path.semantic_similarity +
            weights.get("coverage", 0.0) * path.coverage_score +
            weights.get("prior", 0.0) * path.prior_score
        )
        if not path.sample_paths:
            path.sample_paths = [{'nodes': deepcopy(path.nodes), 'edges': deepcopy(path.edges)}]
        if not path.sample_paths_total_hits:
            path.sample_paths_total_hits = len(path.sample_paths)
        if not path.pattern_structure:
            path.pattern_structure = build_pattern_structure(path)
        if not path.anchor_nodes:
            path.anchor_nodes = extract_anchor_nodes_from_path(path)
        path.anchor_count = len(path.anchor_nodes)
        path.pattern_signature = path.get_signature()

    def _compute_semantic_similarity(self, description: str) -> float:
        if (
            not description
            or self.embedder is None
            or self._normalized_question_embedding is None
        ):
            return 0.0
        try:
            desc_emb = self.embedder.encode_single(description)
            desc_norm = desc_emb / (np.linalg.norm(desc_emb) + 1e-8)
            return float(np.dot(desc_norm, self._normalized_question_embedding))
        except Exception as exc:
            logger.warning("[OptionalMatch] Semantic similarity failed: %s", exc)
            return 0.0

    def _extract_node_snapshot(
        self,
        node_value: Any,
        fallback_label: Optional[str],
        fallback_id: Optional[str]
    ) -> Dict[str, Any]:
        snapshot = {'label': fallback_label, 'id': fallback_id, 'properties': {}}

        if node_value is None:
            return snapshot

        try:
            snapshot['properties'] = dict(node_value)
        except Exception:
            snapshot['properties'] = {}

        try:
            labels = list(node_value.labels)
            if labels:
                snapshot['label'] = labels[0]
        except Exception:
            pass

        try:
            node_id = node_value.get('id')
        except Exception:
            node_id = None
        if node_id is None:
            node_id = getattr(node_value, 'id', fallback_id)
        snapshot['id'] = node_id

        return snapshot

    def _extract_relationship_snapshot(self, rel_value: Any, rel_type: str) -> Dict[str, Any]:
        snapshot = {'type': rel_type, 'properties': {}}

        if rel_value is None:
            return snapshot

        try:
            snapshot['properties'] = dict(rel_value)
        except Exception:
            snapshot['properties'] = {}

        return snapshot


# ============================================================================
# 语义引导的Beam Search（集成平衡方案）
# ============================================================================

class SemanticGuidedBeamSearch:
    """语义引导的Beam Search（集成平衡方案）"""

    def __init__(
        self,
        config: RetrievalConfig,
        schema: Dict,
        question: str,
        question_analysis: Optional[Dict],
        embedder: BGEEmbedder,
        path_scorer: Any,
        relation_index: Optional[Dict] = None,
        description_builder: Optional[PathDescriptionBuilder] = None
    ):
        self.config = config
        self.schema = schema
        self.question = question
        self.question_analysis = question_analysis
        self.embedder = embedder
        self.path_scorer = path_scorer
        self.description_builder = description_builder

        self.candidate_generator = BalancedCandidateGenerator(
            config, schema, question, question_analysis, relation_index,
            description_builder=description_builder
        )

        related_schema = question_analysis.get('related_schema') if question_analysis else None
        self.semantic_filter = SemanticFilter(config, embedder, question, related_schema=related_schema)

        question_embedding = getattr(self.semantic_filter, "question_embedding", None)
        self.executor = OptionalMatchExecutor(
            config,
            embedder=embedder,
            question_embedding=question_embedding,
            description_builder=description_builder,
            related_schema=related_schema
        )

        logger.info(f"[BeamSearch] 初始化完成（平衡方案）")

    def expand(
        self,
        initial_paths: List[PathInstance],
        pool: Any,
        db_name: str
    ) -> List[PathInstance]:
        """执行Beam Search扩展，返回扩展后的路径"""
        beam = initial_paths[:self.config.beam_width]
        all_expanded: List[PathInstance] = []

        logger.info(f"[BeamSearch] 开始扩展")
        logger.info(f"  • 初始路径: {len(beam)}")
        logger.info(f"  • 最大深度: {self.config.max_expansion_depth}")

        for depth in range(self.config.max_expansion_depth):
            logger.info(f"\n[BeamSearch] ===== 深度 {depth+1}/{self.config.max_expansion_depth} =====")
            current_beam = beam[:]
            current_beam_ids = {p.path_id for p in current_beam}

            all_candidates = self.candidate_generator.generate_candidates(beam)
            logger.info(
                "[BeamSearch] 深度%d: 生成候选=%d (beam=%d)",
                depth + 1, len(all_candidates), len(beam)
            )

            if not all_candidates:
                logger.info(f"[BeamSearch] 无候选，停止扩展")
                break

            top_candidates = self.semantic_filter.filter_and_rank(
                all_candidates, top_k=self.config.candidate_width
            )
            logger.info(
                "[BeamSearch] 深度%d: 语义筛选后候选=%d", depth + 1, len(top_candidates)
            )

            if not top_candidates:
                logger.info(f"[BeamSearch] 过滤后无候选，停止扩展")
                break

            base_paths_dict = {p.path_id: p for p in beam}
            expanded_paths = self.executor.execute_candidates(
                top_candidates, base_paths_dict, pool, db_name,
                semantic_filter=self.semantic_filter
            )
            logger.info(
                "[BeamSearch] 深度%d: 执行成功路径=%d", depth + 1, len(expanded_paths)
            )

            if not expanded_paths:
                logger.info(f"[BeamSearch] 无扩展成功，停止")
                break

            all_expanded.extend(expanded_paths)

            candidate_pool = current_beam + expanded_paths
            beam = self._select_top_k_for_next_round(
                candidate_pool, prev_beam_ids=current_beam_ids
            )

            logger.info(f"[BeamSearch] 本轮生成 {len(expanded_paths)} 条路径")
            logger.info(f"[BeamSearch] 下一轮beam: {len(beam)} 条路径")

        logger.info(f"\n[BeamSearch] 扩展完成")
        logger.info(f"  • 总扩展路径: {len(all_expanded)}")

        return all_expanded

    def _select_top_k_for_next_round(
        self,
        paths: List[PathInstance],
        prev_beam_ids: Optional[Set[str]] = None
    ) -> List[PathInstance]:
        """选择top-k路径进入下一轮（保留父路径与子路径的多样性）"""
        prev_ids = prev_beam_ids or set()

        by_signature: Dict[str, List[PathInstance]] = defaultdict(list)
        for path in paths:
            sig = path.get_signature()
            by_signature[sig].append(path)

        selected: List[PathInstance] = []

        def path_score(p: PathInstance) -> float:
            return p.final_score if p.final_score is not None else p.semantic_similarity

        for sig, group in by_signature.items():
            group.sort(key=path_score, reverse=True)
            parents = [p for p in group if p.path_id in prev_ids]
            children = [p for p in group if p.path_id not in prev_ids]

            signature_selected: List[PathInstance] = []
            if parents:
                signature_selected.append(parents[0])
            if children and len(signature_selected) < 2:
                signature_selected.append(children[0])

            for candidate in group:
                if len(signature_selected) >= 2:
                    break
                if candidate not in signature_selected:
                    signature_selected.append(candidate)

            selected.extend(signature_selected)

        selected.sort(key=path_score, reverse=True)
        return selected[:self.config.beam_width]
