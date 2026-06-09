"""
增强型子图检索系统 - 完整测试

版本: 2.0
日期: 2025-12-31
"""

import json
import time
from datetime import datetime
from typing import Dict, Any

from graph_connector import create_pool
from subgraph_retrieval import SubgraphRetriever, RetrievalConfig
from triple_retrieval import EnhancedSubgraphRetriever
from config import SUBGRAPH_DIR, SCHEMA_DIR

print("="*100)
print(" "*35 + "完整功能测试")
print("="*100)
print(f"\n测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*100)


# ============================================================================
# 加载测试用例
# ============================================================================

print("\n" + "="*100)
print("【步骤1】加载测试用例")
print("="*100)

# 测试文件路径（可用 METACYPHER_SUBGRAPH_DIR 覆盖，见 config.py）
test_file = str(SUBGRAPH_DIR / 'case.json')

try:
    with open(test_file, 'r') as f:
        case = json.load(f)
    print(f"\n✅ 测试用例加载成功")
    print(f"  问题: {case.get('question', case.get('nl_question', 'N/A'))}")
    print(f"  数据库: {case.get('graph', 'N/A')}")
except FileNotFoundError:
    print(f"\n❌ 测试用例文件未找到: {test_file}")
    exit(1)
except Exception as e:
    print(f"\n❌ 加载失败: {e}")
    exit(1)


# ============================================================================
# 加载Schema
# ============================================================================

print("\n" + "="*100)
print("【步骤2】加载Schema和实体信息")
print("="*100)

# 从case的analysis中提取schema信息
analysis = case.get('analysis', {})
related_schema = analysis.get('related_schema', {})

# 构建简化schema (从relation_candidates提取)
relation_candidates = related_schema.get('relation_candidates', [])
relationships = []

for rel_cand in relation_candidates:
    relationships.append({
        'from': rel_cand.get('subj_label', ''),
        'to': rel_cand.get('obj_label', ''),
        'type': rel_cand.get('type', '')
    })

# 如果没有relation_candidates，使用默认的geography schema
if not relationships:
    relationships = [
        {'from': 'River', 'to': 'Country', 'type': 'flowsThrough'},
        {'from': 'Lake', 'to': 'Country', 'type': 'locatedIn'},
    ]

schema = {'relationships': relationships}

# 完整schema路径
graph_name = case.get('graph', 'geography')
full_schema_path = f'{SCHEMA_DIR}/{graph_name}.json'

print(f"\n✅ Schema加载完成")
print(f"  简化Schema关系数: {len(schema['relationships'])}")
print(f"  完整Schema路径: {full_schema_path}")
print(f"  节点标签: {related_schema.get('node_labels', [])}")
print(f"  关系类型: {related_schema.get('relation_types', [])}")

# 显示实体信息
entities = analysis.get('entities', [])
print(f"\n实体信息:")
print(f"  识别的实体数: {len(entities)}")
for i, entity in enumerate(entities[:3], 1):
    print(f"  [{i}] {entity.get('mention', 'N/A')} -> {entity.get('normalized', 'N/A')}")
    faiss_cands = entity.get('faiss_candidates', [])
    if faiss_cands:
        best = faiss_cands[0]
        print(f"      最佳匹配: {best.get('node_id', 'N/A')} (distance: {best.get('distance', 0):.4f})")

# 显示retrieved_patterns信息
retrieved_patterns_data = case.get('retrieved_patterns', {})
if isinstance(retrieved_patterns_data, dict):
    # 实际的patterns在'patterns'字段中
    retrieved_patterns = retrieved_patterns_data.get('patterns', [])
else:
    # 如果直接是list（兼容性）
    retrieved_patterns = retrieved_patterns_data if isinstance(retrieved_patterns_data, list) else []

print(f"\n检索到的路径模式:")
print(f"  模式数量: {len(retrieved_patterns)}")
for i, pattern in enumerate(retrieved_patterns[:3], 1):
    print(f"  [{i}] {pattern.get('pattern_id', 'N/A')}")
    print(f"      Pattern: {pattern.get('cypher_pattern', 'N/A')}")
    print(f"      Score: {pattern.get('scores', {}).get('hybrid', 0):.4f}")


# ============================================================================
# 配置检索器
# ============================================================================

print("\n" + "="*100)
print("【步骤3】配置检索器")
print("="*100)

config = RetrievalConfig(
    use_beam_search=True,
    beam_width=20,
    max_expansion_depth=2,
    expand_from_all_nodes=True,
    enable_repetition_detection=True,
    top_k_paths=20,
    debug_print_cypher=True,  # ⭐ 启用Cypher打印
    debug_max_results_display=3,
)

print(f"\n检索配置:")
print(f"  ✅ Beam Search: {config.use_beam_search}")
print(f"  ✅ Beam宽度: {config.beam_width}")
print(f"  ✅ 从所有节点扩展: {config.expand_from_all_nodes}")
print(f"  ✅ 重复关系检测: {config.enable_repetition_detection}")


# ============================================================================
# 测试新版SubgraphRetriever
# ============================================================================

print("\n" + "="*100)
print("【步骤4】测试SubgraphRetriever")
print("="*100)

result_base = None
time_base = 0

try:
    retriever_base = SubgraphRetriever(config, schema, full_schema_path)
    print(f"\n✅ 检索器创建成功")
    
    start_time = time.time()
    with create_pool("cypherbench") as pool:
        result_base = retriever_base.retrieve(pool, case['graph'], case)
    time_base = time.time() - start_time
    
    print(f"\n✅ 检索完成 (耗时: {time_base:.2f}秒)")
    
    stats = result_base.get('stats', {})
    print(f"\n检索结果:")
    print(f"  Pattern路径: {stats.get('pattern_paths_count', 0)}")
    print(f"  Beam扩展路径: {stats.get('beam_paths_count', 0)}")
    print(f"  最终路径: {stats.get('final_paths_count', 0)}")
    
    final_paths = result_base.get('final_paths', [])
    if final_paths:
        print(f"\nTop-3 路径:")
        for i, path in enumerate(final_paths[:3], 1):
            print(f"  [{i}] {path.get_signature()}")
            print(f"      分数: {path.final_score:.4f}")

except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# 测试EnhancedSubgraphRetriever
# ============================================================================

print("\n" + "="*100)
print("【步骤5】测试EnhancedSubgraphRetriever")
print("="*100)

result_enhanced = None
time_enhanced = 0

try:
    retriever_enhanced = EnhancedSubgraphRetriever(
        config=config,
        schema=schema,
        full_schema_path=full_schema_path,  # ⭐ 传递完整schema路径
        enable_semantics=False
    )
    print(f"\n✅ 增强检索器创建成功")
    print(f"  语义生成: {'启用' if retriever_enhanced.enable_semantics else '禁用'}")
    
    start_time = time.time()
    with create_pool("cypherbench") as pool:
        result_enhanced = retriever_enhanced.retrieve_with_triples(pool, case['graph'], case)
    time_enhanced = time.time() - start_time
    
    print(f"\n✅ 检索完成 (耗时: {time_enhanced:.2f}秒)")
    print(f"\n生成三元组数: {result_enhanced.get('num_triples', 0)}")
    
    triples = result_enhanced.get('triples', [])
    if triples:
        print(f"\nTop-3 三元组:")
        for i, triple in enumerate(triples[:3], 1):
            print(f"\n  【三元组 {i}】")
            print(f"    ID: {triple.triple_id}")
            print(f"    Pattern: {triple.cypher_pattern[:80]}...")
            print(f"    节点数: {triple.num_nodes}")
            print(f"    边数: {triple.num_edges}")
            print(f"    分数: {triple.score:.4f}")

except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# 对比分析
# ============================================================================

print("\n" + "="*100)
print("【步骤6】对比分析")
print("="*100)

if result_base and result_enhanced:
    print(f"\n性能对比:")
    print(f"  执行时间: {time_base:.2f}秒 vs {time_enhanced:.2f}秒")
    print(f"  路径数: {len(result_base.get('final_paths', []))} vs {len(result_enhanced.get('final_paths', []))}")
    print(f"  三元组数: N/A vs {result_enhanced.get('num_triples', 0)}")
    
    print(f"\n功能对比:")
    print(f"  平衡扩展: ✅ vs ✅")
    print(f"  Beam Search: ✅ vs ✅")
    print(f"  三元组管理: ❌ vs ✅")


# ============================================================================
# 测试总结
# ============================================================================

print("\n" + "="*100)
print("【步骤7】测试总结")
print("="*100)

success = sum([1 if result_base else 0, 1 if result_enhanced else 0])
print(f"\n成功率: {success}/2 ({100*success/2:.0f}%)")

print(f"\n核心验证:")
print(f"  {'✅' if result_base else '❌'} SubgraphRetriever测试")
print(f"  {'✅' if result_enhanced else '❌'} EnhancedSubgraphRetriever测试")
print(f"  {'✅' if result_enhanced and result_enhanced.get('triples') else '❌'} 三元组生成")

print("\n" + "="*100)
print("测试完成!")
print("="*100)