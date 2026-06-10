#!/usr/bin/env python3
"""
Meta-graph -> Cypher surface realization (paper "From Meta-Graph to Cypher",
subsec:generation).

Implements the generation stage F(Q, M*): the selected meta-graph is serialized
into the LLM prompt as the structural context (the retrieved typed nodes/relations
+ sample paths), and the LLM realizes projection, aggregation, ordering, and syntax.
The structural-context block is the lever ablated by tab:ablation_overall
("w/o structural context", METACYPHER_ABLATE_STRUCTURAL_CONTEXT).

从子图检索结果生成Cypher查询

功能:
1. 读取JSONL文件,提取related_schema和top_triples
2. 构建prompt调用LLM生成Cypher
3. 输出结果到指定JSONL文件

使用方法:
    python generate_cypher_from_subgraph.py \
        --input data/retrieved_subgraphs.jsonl \
        --output data/generated_cypher.jsonl \
        --model "Qwen/Qwen2.5-7B-Instruct" \
        --base_url "http://localhost:8000/v1" \
        --top_k 5 \
        --temperature 0.1
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import sys

# 导入LLM client
from llm_client import SimpleLLMClient
from config import VLLM_BASE_URL, VLLM_MODEL, ABLATE_STRUCTURAL_CONTEXT
try:  # package-style import (pip install -e .)
    from . import instrumentation as _instr
except ImportError:  # flat import, run from inside the package dir
    import instrumentation as _instr


class CypherGenerator:
    """基于子图检索结果生成Cypher查询"""
    
    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        top_k_triples: int = 5
    ):
        """
        初始化生成器
        
        Args:
            model: 模型名称
            base_url: vLLM服务地址
            api_key: API密钥(vLLM通常不需要)
            temperature: 生成温度(低温度更确定性)
            max_tokens: 最大生成token数
            top_k_triples: 使用top-K个三元组作为示例
        """
        self.llm_client = SimpleLLMClient(
            provider="vllm",
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_memory=False  # 每个问题独立生成,不需要记忆
        )
        self.top_k_triples = top_k_triples
        
        # 系统提示(持久化)
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        return """You are an expert in generating Cypher queries for Neo4j graph databases.

Your task is to generate a correct and executable Cypher query based on:
	1.	The user’s natural language question
	2.	The relevant graph schema (node labels and relationship types)
	3.	Retrieved subgraph examples (sample triples with their semantic descriptions)

Important guidelines:
	•	Use only the node labels and relationship types provided in the schema.
	•	Refer to the retrieved subgraph examples to understand the graph structure.
	•	Generate syntactically correct Cypher queries.
	•	Use appropriate WHERE clauses for filtering.
	•	Include ORDER BY if the question asks for sorting.
	•	Include LIMIT if the question asks for top-K results.
	•	Use DISTINCT when necessary to avoid duplicates.

Additional generation rules and strategies:
	1.	Property constraints on nodes
When constraining a node property, prefer specifying the property directly in the node pattern (instead of only using WHERE), for example:
MATCH (n:SomeLabel {{property: <value_placeholder>}})
	2.	Comparative questions (comparison between two entities of the same type)
For questions that compare two entities and choose one based on a property, you may use a CASE expression such as:
MATCH (a:SomeLabel {{property: <value_placeholder>}}),
(b:SomeLabel {{property: <value_placeholder>}})
RETURN CASE
WHEN a.some_numeric_property > b.some_numeric_property THEN a.some_return_property
ELSE b.some_return_property
END AS answer
	3.	Questions requiring a base pattern plus optionally related information
When a relationship may or may not exist, use OPTIONAL MATCH for the optional part, while keeping the required pattern in MATCH, for example:
MATCH (n:SomeLabel)-[:SomeRelation]->(m:AnotherLabel)
OPTIONAL MATCH (n)<-[:AnotherRelation]-(x:OptionalLabel)
RETURN n, m, x
	4.	Temporal questions (who held a role in a specific year)
For time-based role or position questions, treat a relationship as an interval, and constrain it as:
MATCH (p:PersonLabel)<-[r:SomeRoleRelation]-(o:OrgLabel)
WHERE r.start_year <= <year_placeholder>
AND (r.end_year >= <year_placeholder> OR r.end_year IS NULL)
RETURN p.name
	5.	OR-type structural conditions on the same entity type
When the question describes alternative structural conditions joined by OR, use a subquery with UNION and deduplicate afterward, for example:
CALL {
MATCH (n:MainLabel)<-[:RelationTypeA]-(x:ContextLabelA)
RETURN n
UNION
MATCH (n:MainLabel)<-[:RelationTypeB]-(y:ContextLabelB)
RETURN n
}
WITH DISTINCT n
RETURN n.name
"""
    
    def _extract_schema_info(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """提取schema信息"""
        related_schema = item['analysis']['related_schema']
        
        return {
            'node_labels': related_schema.get('node_labels', []),
            'relation_types': related_schema.get('relation_types', [])
        }
    
    def _extract_top_triples(self, item: Dict[str, Any], top_k: int = -1) -> List[Dict[str, Any]]:
        """提取top-K个三元组"""
        if top_k == -1:
            top_k = self.top_k_triples
        
        top_triples = item['subgraph_retrieval_result']['top_triples'][:top_k]
        
        extracted = []
        for triple in top_triples:
            extracted.append({
                'cypher_query': triple['cypher_query'],
                'semantic_description': triple['semantic_description'],
                'sample_paths': triple['sample_paths']
            })
        
        return extracted
    
    def _format_schema_for_prompt(self, schema: Dict[str, Any]) -> str:
        """格式化schema信息为prompt文本"""
        lines = ["## Graph Schema"]
        
        # 节点标签
        if schema['node_labels']:
            lines.append("\n**Node Labels:**")
            for label in schema['node_labels']:
                lines.append(f"- {label}")
        
        # 关系类型
        if schema['relation_types']:
            lines.append("\n**Relationship Types:**")
            for rel_type in schema['relation_types']:
                lines.append(f"- {rel_type}")
        
        return "\n".join(lines)
    
    def _format_triples_for_prompt(self, triples: List[Dict[str, Any]]) -> str:
        """格式化三元组为prompt文本"""
        if not triples:
            return "## Retrieved Subgraph Examples\n\n(No examples available)"
        
        lines = ["## Retrieved Subgraph Examples"]
        lines.append("\nThe following are relevant subgraph patterns retrieved from the database:\n")
        
        for idx, triple in enumerate(triples, 1):
            lines.append(f"### Example {idx}")
            
            # 语义描述
            lines.append(f"**Semantic Description:**")
            lines.append(triple['semantic_description'])
            
            # Cypher查询
            lines.append(f"\n**Cypher Pattern:**")
            lines.append(f"```")
            lines.append(triple['cypher_query'])
            lines.append(f"```")
            
            # Sample paths (可选,如果太长可以省略)
            if triple['sample_paths'] and len(str(triple['sample_paths'])) < 500:
                lines.append(f"\n**Sample Paths:**")
                lines.append(f"```json")
                lines.append(json.dumps(triple['sample_paths'], indent=2, ensure_ascii=False))
                lines.append(f"```")
            
            lines.append("")  # 空行分隔
        
        return "\n".join(lines)
    
    def _build_generation_prompt(
        self,
        question: str,
        schema: Dict[str, Any],
        triples: List[Dict[str, Any]]
    ) -> str:
        """构建完整的生成prompt"""
        parts = []

        # 1. Schema信息
        parts.append(self._format_schema_for_prompt(schema))
        parts.append("")

        # 2. 检索到的三元组 — the "structural context" of tab:ablation_overall:
        #    the selected meta-graph serialized for the final LLM call. Under
        #    METACYPHER_ABLATE_STRUCTURAL_CONTEXT it is omitted (schema and
        #    question are kept).
        if not ABLATE_STRUCTURAL_CONTEXT:
            parts.append(self._format_triples_for_prompt(triples))
            parts.append("")
        
        # 3. 用户问题
        parts.append("## User Question")
        parts.append(f"\n{question}")
        parts.append("")
        
        # 4. 生成指令
        parts.append("## Task")
        parts.append("Generate a Cypher query that answers the above question using the provided schema and examples.")
        parts.append("Output only the Cypher query without any explanations or formatting markers.")
        
        return "\n".join(parts)
    
    def generate_cypher(self, item: Dict[str, Any]) -> str:
        """
        为单个问题生成Cypher查询
        
        Args:
            item: JSONL中的一条数据
            
        Returns:
            生成的Cypher查询
        """
        # 提取信息
        question = item['nl_question']
        schema = self._extract_schema_info(item)
        triples = self._extract_top_triples(item)
        
        # 构建prompt
        user_prompt = self._build_generation_prompt(question, schema, triples)

        # fig:context — record the generation-context length per query (both in
        # the active instrumentation collector and on the item itself, so the
        # output rows carry the x-axis of the EX-vs-context-length figure).
        prompt_chars = len(self.system_prompt) + len(user_prompt)
        prompt_tokens_est = _instr.estimate_tokens(self.system_prompt) + _instr.estimate_tokens(user_prompt)
        _instr.record_prompt(prompt_chars, prompt_tokens_est)
        item['_generation_prompt_chars'] = prompt_chars
        item['_generation_prompt_tokens_est'] = prompt_tokens_est

        # 调用LLM生成
        try:
            cypher_query = self.llm_client.generate(
                system=self.system_prompt,
                user_temp=user_prompt,
                reset_memory=True
            )
            
            # 清理生成结果
            cypher_query = self._clean_generated_cypher(cypher_query)
            
            return cypher_query
            
        except Exception as e:
            print(f"Error generating Cypher for qid={item.get('qid', 'unknown')}: {e}", file=sys.stderr)
            return ""
    
    def _clean_generated_cypher(self, cypher: str) -> str:
        """清理生成的Cypher查询"""
        # 去除markdown代码块标记
        if cypher.startswith("```cypher"):
            cypher = cypher[len("```cypher"):].strip()
        elif cypher.startswith("```"):
            cypher = cypher[3:].strip()
        
        if cypher.endswith("```"):
            cypher = cypher[:-3].strip()
        
        # 去除前后空白
        cypher = cypher.strip()
        
        return cypher
    
    def process_jsonl(
        self,
        input_path: str,
        output_path: str,
        limit: int = 0,
        skip_errors: bool = True
    ):
        """
        批量处理JSONL文件
        
        Args:
            input_path: 输入JSONL文件路径
            output_path: 输出JSONL文件路径
            limit: 最多处理多少条(None表示全部)
            skip_errors: 是否跳过错误继续处理
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 读取输入数据
        print(f"Loading data from {input_path}...")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
        
        if limit:
            data = data[:limit]
        
        print(f"Processing {len(data)} questions...")
        
        # 处理并写入输出
        success_count = 0
        error_count = 0
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in tqdm(data, desc="Generating Cypher"):
                try:
                    # 生成Cypher
                    cypher_query = self.generate_cypher(item)
                    
                    # 构建输出
                    output_item = {
                        'qid': item['qid'],
                        'graph': item['graph'],
                        'question': item['nl_question'],
                        'generated_cypher': cypher_query,
                        'generation_prompt_chars': item.get('_generation_prompt_chars'),
                        'generation_prompt_tokens_est': item.get('_generation_prompt_tokens_est')
                    }
                    
                    # 写入文件
                    f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                    f.flush()  # 立即写入,防止丢失
                    
                    success_count += 1
                    
                except Exception as e:
                    error_count += 1
                    error_msg = f"Error processing qid={item.get('qid', 'unknown')}: {e}"
                    
                    if skip_errors:
                        print(f"\n{error_msg}", file=sys.stderr)
                        # 写入错误记录
                        f.write(json.dumps({
                            'qid': item.get('qid', 'unknown'),
                            'graph': item.get('graph', 'unknown'),
                            'question': item.get('nl_question', ''),
                            'generated_cypher': '',
                            'error': str(e)
                        }, ensure_ascii=False) + '\n')
                        f.flush()
                        continue
                    else:
                        raise RuntimeError(error_msg)
        
        # 打印统计信息
        print(f"\n{'='*60}")
        print(f"Generation completed!")
        print(f"  Success: {success_count}")
        print(f"  Errors:  {error_count}")
        print(f"  Total:   {len(data)}")
        print(f"  Output:  {output_path}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Cypher queries from subgraph retrieval results"
    )
    
    # 输入输出
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input JSONL file with subgraph retrieval results'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output JSONL file for generated Cypher queries'
    )
    
    # LLM配置
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=VLLM_MODEL,
        help='Model name (default from METACYPHER_VLLM_MODEL)'
    )
    parser.add_argument(
        '--base_url', '-u',
        type=str,
        default=VLLM_BASE_URL,
        help='vLLM base URL (default from METACYPHER_VLLM_BASE_URL, e.g. http://localhost:8000/v1)'
    )
    parser.add_argument(
        '--api_key', '-k',
        type=str,
        default='',
        help='API key (usually not needed for vLLM)'
    )
    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=0.1,
        help='Generation temperature (default: 0.1 for deterministic)'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=4000,
        help='Maximum tokens to generate (default: 2048)'
    )
    
    # 生成参数
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='Number of top triples to use as examples (default: 5)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=-1,
        help='Limit number of questions to process (default: all)'
    )
    parser.add_argument(
        '--skip_errors',
        action='store_true',
        help='Skip errors and continue processing'
    )
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = CypherGenerator(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_k_triples=args.top_k
    )
    
    # 处理数据
    generator.process_jsonl(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        skip_errors=args.skip_errors
    )


if __name__ == "__main__":
    main()