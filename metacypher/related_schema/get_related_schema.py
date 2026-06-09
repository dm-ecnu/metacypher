#!/usr/bin/env python3
"""
脚本功能：从JSONL文件中读取问题和分析结果，并补充完整的schema信息
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Default schema dir is env-overridable (METACYPHER_SCHEMA_DIR / PROJECT_ROOT);
# falls back to <repo>/data/schema. See ../config.py.
_DEFAULT_SCHEMA_DIR = Path(os.environ.get("METACYPHER_SCHEMA_DIR") or
                          (Path(os.environ.get("PROJECT_ROOT") or os.environ.get("METACYPHER_DATA_DIR") or
                                (Path(__file__).resolve().parents[2] / "data")) / "schema"))


def load_schema(schema_path: str) -> Dict[str, Any]:
    """
    加载schema文件
    
    Args:
        schema_path: schema文件的路径
        
    Returns:
        schema字典
    """
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_entity_by_label(entities: List[Dict], label: str) -> Optional[Dict]:
    """
    根据label查找实体定义
    
    Args:
        entities: 实体列表
        label: 实体标签
        
    Returns:
        实体定义字典，如果未找到返回None
    """
    for entity in entities:
        if entity['label'] == label:
            return entity
    return None


def find_relation_by_type(relations: List[Dict], rel_type: str, 
                          subj_label: Optional[str] = None, 
                          obj_label: Optional[str] = None) -> Optional[Dict]:
    """
    根据关系类型和可选的主体/客体标签查找关系定义
    
    Args:
        relations: 关系列表
        rel_type: 关系类型
        subj_label: 主体节点标签（可选）
        obj_label: 客体节点标签（可选）
        
    Returns:
        关系定义字典，如果未找到返回None
    """
    for relation in relations:
        if relation['label'] == rel_type:
            # 如果提供了subj_label和obj_label，则精确匹配
            if subj_label and obj_label:
                if relation['subj_label'] == subj_label and relation['obj_label'] == obj_label:
                    return relation
            else:
                # 否则只匹配关系类型
                return relation
    return None


def enrich_related_schema(analysis: Dict, schema: Dict) -> Dict:
    """
    根据analysis中的related_schema信息，从完整schema中提取详细信息
    
    Args:
        analysis: 包含related_schema的分析结果
        schema: 完整的schema定义
        
    Returns:
        enriched_schema: 包含详细信息和confidence/reason的schema
    """
    related_schema = analysis.get('related_schema', {})
    
    enriched = {
        'entities': [],
        'relations': []
    }
    
    # 处理节点标签候选
    node_candidates = related_schema.get('node_label_candidates', [])
    for candidate in node_candidates:
        label = candidate.get('label')
        confidence = candidate.get('confidence')
        reason = candidate.get('reason')
        
        # 从schema中查找完整的实体定义
        entity_def = find_entity_by_label(schema['entities'], label)
        
        if entity_def:
            enriched_entity = {
                'label': entity_def['label'],
                'description': entity_def.get('description'),
                'properties': entity_def.get('properties', {}),
                'confidence': confidence,
                'reason': reason
            }
            enriched['entities'].append(enriched_entity)
    
    # 处理关系候选
    relation_candidates = related_schema.get('relation_candidates', [])
    for candidate in relation_candidates:
        rel_type = candidate.get('type')
        subj_label = candidate.get('subj_label')
        obj_label = candidate.get('obj_label')
        direction = candidate.get('direction')
        pattern = candidate.get('pattern')
        confidence = candidate.get('confidence')
        reason = candidate.get('reason')
        
        # 从schema中查找完整的关系定义
        relation_def = find_relation_by_type(
            schema['relations'], 
            rel_type, 
            subj_label, 
            obj_label
        )
        
        if relation_def:
            enriched_relation = {
                'label': relation_def['label'],
                'subj_label': relation_def['subj_label'],
                'obj_label': relation_def['obj_label'],
                'properties': relation_def.get('properties', {}),
                'pattern': relation_def.get('pattern'),
                'direction': direction,
                'confidence': confidence,
                'reason': reason
            }
            enriched['relations'].append(enriched_relation)
    
    return enriched


def process_jsonl(input_file: str, output_file: str, schema_dir: str):
    """
    处理JSONL文件，为每一行补充完整的schema信息
    
    Args:
        input_file: 输入JSONL文件路径
        output_file: 输出JSONL文件路径
        schema_dir: schema文件所在目录
    """
    schema_cache = {}  # 缓存已加载的schema
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # 解析JSON行
                data = json.loads(line.strip())
                
                # 提取必要字段
                qid = data.get('qid')
                graph = data.get('graph')
                nl_question = data.get('nl_question')
                analysis = data.get('analysis', {})
                
                # 检查必要字段是否存在
                if not all([qid, graph, nl_question]):
                    print(f"Warning: Line {line_num} missing required fields, skipping...")
                    continue
                
                # 加载对应的schema文件（使用缓存）
                if graph not in schema_cache:
                    schema_path = os.path.join(schema_dir, f"{graph}.json")
                    if not os.path.exists(schema_path):
                        print(f"Warning: Schema file not found for graph '{graph}': {schema_path}")
                        continue
                    schema_cache[graph] = load_schema(schema_path)
                
                schema = schema_cache[graph]
                
                # 补充schema信息
                related_schema = enrich_related_schema(analysis, schema)
                
                # 构建输出记录
                output_record = {
                    'qid': qid,
                    'graph': graph,
                    'nl_question': nl_question,
                    'related_schema': related_schema
                }
                
                # 写入输出文件
                outfile.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                
                if line_num % 100 == 0:
                    print(f"Processed {line_num} lines...")
                    
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON at line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error: Failed to process line {line_num}: {e}")
                continue
    
    print(f"Processing complete. Output written to {output_file}")


def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enrich JSONL file with complete schema information'
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input JSONL file path'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output JSONL file path'
    )
    parser.add_argument(
        '-s', '--schema-dir',
        default=str(_DEFAULT_SCHEMA_DIR),
        help='Directory containing schema JSON files (default: %(default)s; env-overridable via METACYPHER_SCHEMA_DIR / PROJECT_ROOT)'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # 检查schema目录是否存在
    if not os.path.exists(args.schema_dir):
        print(f"Error: Schema directory not found: {args.schema_dir}")
        return
    
    # 处理文件
    process_jsonl(args.input, args.output, args.schema_dir)


if __name__ == '__main__':
    main()