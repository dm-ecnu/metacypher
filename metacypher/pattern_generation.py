"""
基于模板的语义生成器 - Template-Based Semantic Generator

功能: 从Cypher模式生成自然语言描述,使用预定义的模板
特点: 
  - 支持match_template和optional_match_template
  - 处理节点属性约束 (name, 其他属性)
  - 处理WHERE子句约束
"""

import json
import re
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

from config import TEMPLATE_DIR, SCHEMA_DIR

logger = logging.getLogger(__name__)

# Default location for template-annotated schema files (env-overridable via config).
DEFAULT_TEMPLATE_BASE_PATH = str(TEMPLATE_DIR)


class TemplateBasedSemanticGenerator:
    """基于模板的语义生成器"""
    
    def __init__(self, template_path: Optional[str] = None):
        """
        初始化生成器
        
        Args:
            template_path: schema_with_template文件路径
        """
        self.templates = {}
        self.property_templates = {
            # 属性操作符的自然语言模板
            '=': '{label} with {property} equals {value}',
            '>': '{label} with {property} greater than {value}',
            '<': '{label} with {property} less than {value}',
            '>=': '{label} with {property} at least {value}',
            '<=': '{label} with {property} at most {value}',
            '!=': '{label} with {property} not equals {value}',
            'IS NULL': '{label} without {property}',
            'IS NOT NULL': '{label} with {property}',
        }
        
        if template_path:
            self.templates = self._load_templates(template_path)
            logger.info(f"[TemplateGenerator] 加载模板: {len(self.templates)} 个关系")
        else:
            logger.warning("[TemplateGenerator] 未加载模板文件,使用默认模板")
    
    def _load_templates(self, template_path: str) -> Dict:
        """
        加载schema_with_template文件
        
        Returns:
            {
                'relation_type': {
                    'subj_label': str,
                    'obj_label': str,
                    'match_template': str,
                    'optional_match_template': str,
                    'pattern': str
                }
            }
        """
        templates = {}
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            
            # 提取关系模板
            for rel in schema.get('relations', []):
                key = f"{rel['subj_label']}_{rel['label']}_{rel['obj_label']}"
                templates[key] = {
                    'label': rel['label'],
                    'subj_label': rel['subj_label'],
                    'obj_label': rel['obj_label'],
                    'match_template': rel.get('match_template', ''),
                    'optional_match_template': rel.get('optional_match_template', ''),
                    'pattern': rel.get('pattern', '')
                }
            
            logger.info(f"[TemplateGenerator] 成功加载 {len(templates)} 个关系模板")
            return templates
        
        except Exception as e:
            logger.error(f"[TemplateGenerator] 模板加载失败: {e}")
            return {}
    
    def generate_semantic(
        self,
        cypher_pattern: str,
        db_name: Optional[str] = None
    ) -> str:
        """
        从Cypher模式生成自然语言描述
        
        Args:
            cypher_pattern: 完整的Cypher模式
                例: "MATCH (n:Lake)-[r0:locatedIn]->(m0:Country)<-[r1:flowsThrough]-(m1:River {name: 'Natara'})
                     WITH DISTINCT n WHERE n.area_km2 < 390000"
            db_name: 数据库名称 (用于动态加载模板)
        
        Returns:
            自然语言描述
        """
        try:
            # 动态加载模板 (如果指定了db_name)
            if db_name and not self.templates:
                self._try_load_template_for_db(db_name)
            
            # 1. 分离MATCH和WHERE子句
            match_clauses, where_clause = self._split_match_where(cypher_pattern)
            
            # 2. 解析每个MATCH子句
            semantic_parts = []
            
            for match_clause in match_clauses:
                is_optional = 'OPTIONAL' in match_clause.upper()
                
                # 解析pattern
                pattern_segments = self._parse_pattern(match_clause)
                
                # 生成每个关系的语义
                for segment in pattern_segments:
                    if segment['type'] == 'relation':
                        semantic = self._generate_relation_semantic(segment, is_optional)
                        if semantic:
                            # 确保以句号结尾
                            if not semantic.endswith('.'):
                                semantic += '.'
                            semantic_parts.append(semantic)
            
            # 3. 处理WHERE约束
            if where_clause:
                where_semantics = self._generate_where_semantics(where_clause)
                for ws in where_semantics:
                    # 首字母大写,添加句号
                    ws = ws[0].upper() + ws[1:] if ws else ws
                    if ws and not ws.endswith('.'):
                        ws += '.'
                    semantic_parts.append(ws)
            
            # 4. 组合
            full_semantic = ' '.join(semantic_parts)
            
            logger.debug(f"[TemplateGenerator] 生成语义: {full_semantic}")
            
            return full_semantic
        
        except Exception as e:
            logger.error(f"[TemplateGenerator] 语义生成失败: {e}")
            # 降级: 返回简化版本
            return self._generate_fallback_semantic(cypher_pattern)
    
    def _try_load_template_for_db(self, db_name: str):
        """尝试动态加载指定数据库的模板"""
        possible_paths = [
            f"{TEMPLATE_DIR}/{db_name}.json",
            f"{SCHEMA_DIR}/template/{db_name}_with_template.json",
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                logger.info(f"[TemplateGenerator] 动态加载模板: {path}")
                self.templates = self._load_templates(path)
                return
        
        logger.warning(f"[TemplateGenerator] 未找到 {db_name} 的模板文件")
    
    def _split_match_where(self, cypher: str) -> Tuple[List[str], str]:
        """
        分离MATCH和WHERE子句
        
        Returns:
            (match_clauses, where_clause)
        """
        # 移除多余空白
        cypher = ' '.join(cypher.split())
        
        # 找到所有MATCH (包括OPTIONAL MATCH)
        match_clauses = []
        
        # 检查是否有MATCH关键字
        has_match_keyword = re.search(r'\b(?:OPTIONAL\s+)?MATCH\b', cypher, re.IGNORECASE)
        
        if has_match_keyword:
            # 正常情况：有MATCH关键字
            # 正则: MATCH ... (直到遇到MATCH/WHERE/WITH/RETURN)
            pattern = r'((?:OPTIONAL\s+)?MATCH\s+.*?)(?=\s+(?:MATCH|OPTIONAL\s+MATCH|WHERE|WITH|RETURN)|$)'
            matches = re.finditer(pattern, cypher, re.IGNORECASE)
            
            for m in matches:
                match_clauses.append(m.group(1).strip())
        else:
            # 特殊情况：没有MATCH关键字，直接把pattern部分作为MATCH
            # 例如: "(n:Lake)-[:locatedIn]->(c:Country)"
            pattern_part = re.split(r'\s+WHERE\s+', cypher, flags=re.IGNORECASE)[0]
            if pattern_part.strip():
                match_clauses.append(f"MATCH {pattern_part.strip()}")
        
        # 提取WHERE子句
        where_match = re.search(r'WHERE\s+(.*?)(?=\s+(?:RETURN|WITH|ORDER|LIMIT)|$)', cypher, re.IGNORECASE)
        where_clause = where_match.group(1).strip() if where_match else ''
        
        return match_clauses, where_clause
    
    def _parse_pattern(self, match_clause: str) -> List[Dict]:
        """
        解析MATCH子句中的pattern
        
        Returns:
            List of {
                'type': 'node' | 'relation',
                'variable': str,
                'label': str,
                'properties': Dict,
                'direction': 'out' | 'in' | 'both'
            }
        """
        # 移除MATCH关键字
        pattern = re.sub(r'^\s*(?:OPTIONAL\s+)?MATCH\s+', '', match_clause, flags=re.IGNORECASE)
        
        segments = []
        
        # 节点模式: (var:Label {props})
        node_pattern = r'\(([a-zA-Z0-9_]*):([a-zA-Z0-9_]+)(?:\s*\{([^}]+)\})?\)'
        
        # 关系模式
        rel_out = r'-\[([a-zA-Z0-9_]*):([a-zA-Z0-9_]+)(?:\s*\{([^}]+)\})?\]->'
        rel_in = r'<-\[([a-zA-Z0-9_]*):([a-zA-Z0-9_]+)(?:\s*\{([^}]+)\})?\]-'
        
        # 找到所有匹配
        pos = 0
        nodes = []
        relations = []
        
        while pos < len(pattern):
            # 尝试匹配节点
            node_match = re.match(node_pattern, pattern[pos:])
            if node_match:
                var, label, props = node_match.groups()
                nodes.append({
                    'type': 'node',
                    'variable': var or '',
                    'label': label,
                    'properties': self._parse_properties(props) if props else {},
                    'pos': pos
                })
                pos += node_match.end()
                continue
            
            # 尝试匹配出边
            rel_match = re.match(rel_out, pattern[pos:])
            if rel_match:
                var, rel_type, props = rel_match.groups()
                relations.append({
                    'type': 'relation',
                    'variable': var or '',
                    'rel_type': rel_type,
                    'properties': self._parse_properties(props) if props else {},
                    'direction': 'out',
                    'pos': pos
                })
                pos += rel_match.end()
                continue
            
            # 尝试匹配入边
            rel_match = re.match(rel_in, pattern[pos:])
            if rel_match:
                var, rel_type, props = rel_match.groups()
                relations.append({
                    'type': 'relation',
                    'variable': var or '',
                    'rel_type': rel_type,
                    'properties': self._parse_properties(props) if props else {},
                    'direction': 'in',
                    'pos': pos
                })
                pos += rel_match.end()
                continue
            
            pos += 1
        
        # 组合节点和关系
        segments = []
        for rel in relations:
            # 找到关系前后的节点
            before_nodes = [n for n in nodes if n['pos'] < rel['pos']]
            after_nodes = [n for n in nodes if n['pos'] > rel['pos']]
            
            if before_nodes and after_nodes:
                subj_node = before_nodes[-1]
                obj_node = after_nodes[0]
                
                segments.append({
                    'type': 'relation',
                    'rel_type': rel['rel_type'],
                    'rel_properties': rel['properties'],
                    'direction': rel['direction'],
                    'subj_node': subj_node,
                    'obj_node': obj_node
                })
        
        return segments
    
    def _parse_properties(self, prop_str: str) -> Dict:
        """
        解析属性字符串
        
        Args:
            prop_str: "name: 'Natara', age: 30"
        
        Returns:
            {'name': 'Natara', 'age': 30}
        """
        properties = {}
        
        # 简单解析: key: value
        pairs = prop_str.split(',')
        for pair in pairs:
            if ':' in pair:
                key, value = pair.split(':', 1)
                key = key.strip()
                value = value.strip().strip("'\"")
                properties[key] = value
        
        return properties
    
    def _generate_relation_semantic(
        self,
        segment: Dict,
        is_optional: bool = False
    ) -> str:
        """
        生成关系的自然语言描述
        
        Args:
            segment: {
                'rel_type': str,
                'subj_node': {label, properties},
                'obj_node': {label, properties},
                'direction': 'out' | 'in'
            }
            is_optional: 是否是OPTIONAL MATCH
        
        Returns:
            自然语言描述
        """
        rel_type = segment['rel_type']
        subj_node = segment['subj_node']
        obj_node = segment['obj_node']
        direction = segment['direction']
        
        # 根据方向调整主客体
        if direction == 'in':
            subj_node, obj_node = obj_node, subj_node
        
        # 查找模板
        template_key = f"{subj_node['label']}_{rel_type}_{obj_node['label']}"
        template_info = self.templates.get(template_key)
        
        if template_info:
            # 使用预定义模板
            template = template_info['optional_match_template'] if is_optional else template_info['match_template']
            
            # 填充占位符 (传入template用于智能识别)
            subj_name = self._get_node_reference(subj_node, template)
            obj_name = self._get_node_reference(obj_node, template)
            
            semantic = template.format(subj_name=subj_name, obj_name=obj_name)
            
            # 清理多余空格（当name为空时可能出现）
            semantic = ' '.join(semantic.split())
        else:
            # 降级: 使用默认模板（简化版）
            subj_name = self._get_node_reference(subj_node, "")
            obj_name = self._get_node_reference(obj_node, "")
            
            # 根据关系类型生成更自然的默认语义
            rel_semantic = self._make_relation_semantic(rel_type)
            
            if is_optional:
                semantic = f"{subj_name} may {rel_semantic} {obj_name}"
            else:
                semantic = f"{subj_name} {rel_semantic} {obj_name}"
        
        return semantic
    
    def _get_node_reference(self, node: Dict, template: str = "") -> str:
        """
        获取节点引用字符串 (用于填充模板)
        
        智能识别模板格式:
        - 如果模板已包含label (如"The lake {subj_name}") → 只返回具体值或属性
        - 如果模板不含label (如"{subj_name} is located") → 返回完整引用
        
        规则:
        1. 有name属性:
           - 模板含label: 只返回name值 "Amazon"
           - 模板不含label: 返回"the label Name"
        2. 有其他属性:
           - 总是返回"with property=value"
        3. 无属性:
           - 模板含label: 返回空字符串
           - 模板不含label: 返回"the label"
        
        Args:
            node: {'label': str, 'properties': Dict}
            template: 模板字符串（用于判断是否已包含label）
        
        Returns:
            节点引用字符串
        """
        label = node['label'].lower()
        props = node.get('properties', {})
        
        # 检测模板是否已包含label
        template_has_label = self._template_contains_label(template, label)
        
        if not props:
            # 无属性
            if template_has_label:
                # 模板已有label，返回空（避免重复）
                return ""
            else:
                # 模板无label，返回完整引用
                return f"the {label}"
        
        if 'name' in props:
            # 有name属性
            name_value = props['name']
            if template_has_label:
                # 模板已有label，只返回name值
                return name_value
            else:
                # 模板无label，返回完整引用
                return f"the {label} {name_value}"
        
        # 有其他属性，总是返回"with ..."格式
        prop_parts = []
        for key, value in props.items():
            formatted_key = self._format_property_name(key)
            prop_parts.append(f"{formatted_key} {value}")
        
        prop_str = " and ".join(prop_parts)
        
        if template_has_label:
            return f"with {prop_str}"
        else:
            return f"the {label} with {prop_str}"
    
    def _make_relation_semantic(self, rel_type: str) -> str:
        """
        将关系类型转换为自然语言动词短语
        
        例如:
        - locatedIn → "is located in"
        - flowsThrough → "flows through"
        - PART_OF → "is part of"
        - KNOWS → "knows"
        - VISITS → "visits"
        
        Args:
            rel_type: 关系类型标签
        
        Returns:
            自然语言动词短语
        """
        # 常见关系类型映射
        relation_map = {
            'locatedIn': 'is located in',
            'flowsThrough': 'flows through',
            'PART_OF': 'is part of',
            'partOf': 'is part of',
            'basedIn': 'is based in',
            'foundedBy': 'was founded by',
            'hasMember': 'has member',
            'belongsTo': 'belongs to',
            'connectedTo': 'is connected to',
            'relatedTo': 'is related to',
            'worksAt': 'works at',
            'livesIn': 'lives in',
            'KNOWS': 'knows',
            'VISITS': 'visits',
            'WORKS_AT': 'works at',
            'LIVES_IN': 'lives in',
        }
        
        # 查找映射
        if rel_type in relation_map:
            return relation_map[rel_type]
        
        # 自动转换
        
        # 1. 处理全大写+下划线格式 (如 PART_OF, WORKS_AT)
        if '_' in rel_type and rel_type.isupper():
            result = rel_type.lower().replace('_', ' ')
            # 添加"is"如果合适
            if not result.startswith(('is ', 'was ', 'has ', 'have ')):
                result = f"is {result}"
            return result
        
        # 2. 处理全大写无下划线格式 (如 KNOWS, VISITS)
        if rel_type.isupper() and '_' not in rel_type:
            return rel_type.lower()
        
        # 3. 处理驼峰命名 (如 flowsThrough, locatedIn)
        if any(c.isupper() for c in rel_type) and not rel_type.isupper():
            # 在大写字母前插入空格，但不处理第一个字符
            result = rel_type[0].lower()
            for c in rel_type[1:]:
                if c.isupper():
                    result += ' ' + c.lower()
                else:
                    result += c
            
            # 添加 "is" 如果需要
            if not result.startswith(('is ', 'was ', 'has ', 'have ', 'flows ', 'knows ', 'visits ')):
                result = f"is {result}"
            
            return result
        
        # 4. 默认：小写 + "is related to by"
        return f"is related to by {rel_type.lower()}"
    
    def _template_contains_label(self, template: str, label: str) -> bool:
        """
        检测模板是否已包含label
        
        例如:
        - "The lake {subj_name}" → True (包含"lake")
        - "{subj_name} is located" → False
        """
        if not template:
            return False
        
        # 转小写比较
        template_lower = template.lower()
        label_lower = label.lower()
        
        # 检测常见模式
        patterns = [
            f"the {label_lower}",
            f"a {label_lower}",
            f"{label_lower} {{",  # "lake {subj_name}"
        ]
        
        for pattern in patterns:
            if pattern in template_lower:
                return True
        
        return False
    
    def _generate_where_semantics(self, where_clause: str) -> List[str]:
        """
        生成WHERE子句的自然语言描述
        
        Args:
            where_clause: "n.area_km2 < 390000 AND m.status = 'active'"
        
        Returns:
            自然语言列表
        """
        semantics = []
        
        # 分割AND/OR
        conditions = re.split(r'\s+AND\s+|\s+OR\s+', where_clause, flags=re.IGNORECASE)
        
        for condition in conditions:
            condition = condition.strip()
            
            # 解析条件: var.property operator value
            match = re.match(r'([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\s*([><=!]+|IS\s+(?:NOT\s+)?NULL)\s*(.+)?', condition, re.IGNORECASE)
            
            if match:
                var, prop, operator, value = match.groups()
                
                # 清理value
                if value:
                    value = value.strip().strip("'\"")
                
                # 生成自然语言
                semantic = self._generate_condition_semantic(prop, operator, value)
                semantics.append(semantic)
            else:
                # 无法解析,直接添加
                semantics.append(f"with condition {condition}")
        
        return semantics
    
    def _generate_condition_semantic(
        self,
        property_name: str,
        operator: str,
        value: Optional[str]
    ) -> str:
        """
        生成单个条件的自然语言
        
        Args:
            property_name: 属性名 (如 'area_km2')
            operator: 操作符 (如 '<')
            value: 值 (如 '390000')
        
        Returns:
            自然语言描述
        """
        # 格式化属性名 (area_km2 → area in km²)
        formatted_prop = self._format_property_name(property_name)
        
        # 格式化值
        formatted_value = self._format_value(value, property_name) if value else ''
        
        # 操作符映射
        operator_map = {
            '<': f'with {formatted_prop} less than {formatted_value}',
            '>': f'with {formatted_prop} greater than {formatted_value}',
            '<=': f'with {formatted_prop} at most {formatted_value}',
            '>=': f'with {formatted_prop} at least {formatted_value}',
            '=': f'with {formatted_prop} equals {formatted_value}',
            '!=': f'with {formatted_prop} not equals {formatted_value}',
            'IS NULL': f'without {formatted_prop}',
            'IS NOT NULL': f'with {formatted_prop}',
        }
        
        semantic = operator_map.get(operator.upper().replace(' ', '_'), f'with {formatted_prop} {operator} {formatted_value}')
        
        return semantic
    
    def _format_property_name(self, prop_name: str) -> str:
        """
        格式化属性名
        
        例: area_km2 → area in square kilometers
        """
        # 常见模式
        replacements = {
            '_km2': ' in square kilometers',
            '_km': ' in kilometers',
            '_m': ' in meters',
            '_usd': ' in USD',
            '_seconds': ' in seconds',
            '_years': ' in years',
            '_': ' ',
        }
        
        formatted = prop_name
        for pattern, replacement in replacements.items():
            if pattern in formatted:
                formatted = formatted.replace(pattern, replacement)
                break
        
        return formatted
    
    def _format_value(self, value: str, property_name: str) -> str:
        """格式化值"""
        # 如果是数字,添加单位
        try:
            num = float(value)
            # 根据属性名推断单位
            if 'km2' in property_name or 'area' in property_name:
                return f"{num}"
            return value
        except:
            return value
    
    def _generate_fallback_semantic(self, pattern: str) -> str:
        """降级语义生成"""
        # 提取标签
        labels = re.findall(r':([a-zA-Z0-9_]+)', pattern)
        if labels:
            return " connected to ".join(labels)
        return "graph pattern"


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("基于模板的语义生成器测试")
    print("="*80)
    
    # 创建测试模板
    print("\n准备测试模板...")
    import json
    test_template = {
        "name": "test",
        "relations": [
            {
                "label": "locatedIn",
                "subj_label": "Lake",
                "obj_label": "Country",
                "match_template": "The lake {subj_name} is located in the country {obj_name}",
                "optional_match_template": "The lake {subj_name} may be located in the country {obj_name}"
            },
            {
                "label": "flowsThrough",
                "subj_label": "River",
                "obj_label": "Country",
                "match_template": "The river {subj_name} flows through the country {obj_name}",
                "optional_match_template": "The river {subj_name} may flow through the country {obj_name}"
            }
        ]
    }
    
    with open('/tmp/test_template.json', 'w') as f:
        json.dump(test_template, f)
    
    # 创建生成器 (加载模板)
    generator = TemplateBasedSemanticGenerator('/tmp/test_template.json')
    
    # 测试用例
    test_cases = [
        {
            'name': '基础MATCH',
            'pattern': "MATCH (n:Lake)-[r0:locatedIn]->(m0:Country)<-[r1:flowsThrough]-(m1:River {name: 'Natara'})"
        },
        {
            'name': 'MATCH + WHERE',
            'pattern': "MATCH (n:Lake)-[r0:locatedIn]->(m0:Country)<-[r1:flowsThrough]-(m1:River {name: 'Natara'}) WITH DISTINCT n WHERE n.area_km2 < 390000"
        },
        {
            'name': 'OPTIONAL MATCH',
            'pattern': "OPTIONAL MATCH (p:Person)-[:KNOWS]->(p2:Person)"
        },
        {
            'name': '多个MATCH',
            'pattern': "MATCH (p:Person {status: 'Sick'})-[v:VISITS]->(pl:Place) MATCH (pl)-[:PART_OF]->(:Region)"
        },
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n【测试 {i}】{test['name']}")
        print(f"Pattern:")
        print(f"  {test['pattern']}")
        
        semantic = generator.generate_semantic(test['pattern'])
        print(f"Semantic:")
        print(f"  {semantic}")
        print("-"*80)
    
    print("\n✅ 测试完成")
    print("\n说明:")
    print("1. 使用测试模板文件")
    print("2. 支持没有MATCH关键字的pattern")
    print("3. 改进了默认模板（无冗余）")
    print("4. 支持的功能:")
    print("   - name属性识别")
    print("   - 其他属性列出")
    print("   - WHERE约束转换")
    print("   - OPTIONAL MATCH识别")


"""
模板语义生成接口 - Simple API

提供简洁的函数接口，输入graph名称和pattern，返回自然语言描述
"""

import json
import logging
from pathlib import Path
from typing import Optional
logger = logging.getLogger(__name__)


# ============================================================================
# 全局缓存
# ============================================================================
_GENERATOR_CACHE = {}  # {graph_name: TemplateBasedSemanticGenerator}


def generate_semantic_from_pattern(
    graph_name: str,
    pattern: str,
    template_base_path: str = DEFAULT_TEMPLATE_BASE_PATH,
    use_cache: bool = True
) -> str:
    """
    从Cypher pattern生成自然语言描述
    
    Args:
        graph_name: 图数据库名称（如 'geography', 'company', 'art'等）
        pattern: 完整的Cypher pattern字符串
            例: "MATCH (n:Lake)-[r0:locatedIn]->(m0:Country) WHERE n.area_km2 < 390000"
        template_base_path: 模板文件基础路径，默认为标准路径
        use_cache: 是否缓存生成器（默认True，避免重复加载）
    
    Returns:
        自然语言描述字符串
        
    Example:
        >>> semantic = generate_semantic_from_pattern(
        ...     graph_name='geography',
        ...     pattern="MATCH (n:Lake)-[r0:locatedIn]->(m0:Country) WHERE n.area_km2 < 390000"
        ... )
        >>> print(semantic)
        "the lake is located in the country. With area in square kilometers less than 390000.0."
    """
    try:
        # 1. 获取或创建生成器
        generator = _get_generator(graph_name, template_base_path, use_cache)
        
        # 2. 生成语义
        semantic = generator.generate_semantic(pattern)
        
        logger.info(f"[API] 成功生成语义: {graph_name}")
        return semantic
    
    except Exception as e:
        logger.error(f"[API] 语义生成失败: {e}")
        # 返回降级结果
        return _generate_fallback(pattern)


def _get_generator(
    graph_name: str,
    template_base_path: str,
    use_cache: bool
) -> TemplateBasedSemanticGenerator:
    """获取或创建语义生成器"""
    
    # 检查缓存
    if use_cache and graph_name in _GENERATOR_CACHE:
        logger.debug(f"[API] 使用缓存的生成器: {graph_name}")
        return _GENERATOR_CACHE[graph_name]
    
    # 构建模板路径
    template_path = Path(template_base_path) / f"{graph_name}.json"
    
    # 检查文件是否存在
    if not template_path.exists():
        logger.warning(f"[API] 模板文件不存在: {template_path}")
        logger.warning(f"[API] 将使用默认模板")
        generator = TemplateBasedSemanticGenerator()
    else:
        logger.info(f"[API] 加载模板文件: {template_path}")
        generator = TemplateBasedSemanticGenerator(str(template_path))
    
    # 缓存
    if use_cache:
        _GENERATOR_CACHE[graph_name] = generator
    
    return generator


def _generate_fallback(pattern: str) -> str:
    """降级语义生成"""
    import re
    labels = re.findall(r':([a-zA-Z0-9_]+)', pattern)
    if labels:
        return " connected to ".join(labels).lower() + "."
    return "graph pattern"


def clear_cache():
    """清空生成器缓存"""
    global _GENERATOR_CACHE
    _GENERATOR_CACHE.clear()
    logger.info("[API] 缓存已清空")


def preload_templates(graph_names: list, template_base_path: str = DEFAULT_TEMPLATE_BASE_PATH):
    """
    预加载多个图的模板
    
    Args:
        graph_names: 图名称列表，如 ['geography', 'company', 'art']
        template_base_path: 模板文件基础路径
    """
    logger.info(f"[API] 预加载 {len(graph_names)} 个模板")
    
    for graph_name in graph_names:
        try:
            _get_generator(graph_name, template_base_path, use_cache=True)
            logger.info(f"[API] ✓ {graph_name}")
        except Exception as e:
            logger.error(f"[API] ✗ {graph_name}: {e}")


def generate_semantic_batch(
    patterns: list,
    graph_name: str,
    template_base_path: str = DEFAULT_TEMPLATE_BASE_PATH
) -> list:
    """
    批量生成语义描述
    
    Args:
        patterns: pattern列表
        graph_name: 图数据库名称
        template_base_path: 模板文件基础路径
    
    Returns:
        语义描述列表
    """
    logger.info(f"[API] 批量生成: {len(patterns)} 个patterns")
    
    # 获取生成器
    generator = _get_generator(graph_name, template_base_path, use_cache=True)
    
    # 批量生成
    results = []
    for i, pattern in enumerate(patterns, 1):
        try:
            semantic = generator.generate_semantic(pattern)
            results.append(semantic)
        except Exception as e:
            logger.error(f"[API] Pattern {i} 失败: {e}")
            results.append(_generate_fallback(pattern))
    
    return results


# ============================================================================
# 便捷函数
# ============================================================================

def quick_generate(graph: str, pattern: str) -> str:
    """
    快速生成（简化版，使用默认路径和缓存）
    
    Args:
        graph: 图名称
        pattern: Cypher pattern
    
    Returns:
        自然语言描述
    """
    return generate_semantic_from_pattern(graph, pattern)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    
    print("="*80)
    print("模板语义生成接口测试")
    print("="*80)
    
    # 测试1: 基础使用
    print("\n【测试1】基础使用")
    
    pattern1 = "MATCH (n:Lake)-[r0:locatedIn]->(m0:Country)<-[r1:flowsThrough]-(m1:River {name: 'Natara'}) WHERE n.area_km2 < 390000"
    
    semantic1 = generate_semantic_from_pattern('geography', pattern1)
    print(f"Pattern:")
    print(f"  {pattern1}")
    print(f"Semantic:")
    print(f"  {semantic1}")
    
    # 测试2: 使用本地模板文件
    print("\n【测试2】使用本地模板")
    
    pattern2 = "MATCH (c:Company {name: 'Google'})-[:operatesIn]->(i:Industry)"
    
    semantic2 = generate_semantic_from_pattern(
        graph_name='company',
        pattern=pattern2
    )
    print(f"Pattern:")
    print(f"  {pattern2}")
    print(f"Semantic:")
    print(f"  {semantic2}")
    
    # 测试3: 批量生成
    print("\n【测试3】批量生成")
    
    patterns = [
        "MATCH (l:Lake)-[:locatedIn]->(c:Country)",
        "MATCH (r:River {name: 'Amazon'})-[:flowsThrough]->(c:Country)",
        "OPTIONAL MATCH (m:Mountain)-[:locatedIn]->(c:Country)"
    ]
    
    semantics = generate_semantic_batch(
        patterns=patterns,
        graph_name='geography'
    )
    
    for i, (pattern, semantic) in enumerate(zip(patterns, semantics), 1):
        print(f"\n[{i}] Pattern: {pattern}")
        print(f"    Semantic: {semantic}")
    
    # 测试4: 预加载
    print("\n【测试4】预加载模板")
    clear_cache()
    preload_templates(['geography', 'company'])
    
    # 测试5: 快速生成
    print("\n【测试5】快速生成")
    semantic5 = quick_generate('geography', "MATCH (l:Lake)-[:locatedIn]->(c:Country)")
    print(f"Semantic: {semantic5}")
    
    print("\n" + "="*80)
    print("✅ 所有测试完成!")
    print("="*80)
    
    print("\n使用说明:")
    print("  1. 主接口: generate_semantic_from_pattern(graph_name, pattern)")
    print("  2. 快速版: quick_generate(graph, pattern)")
    print("  3. 批量版: generate_semantic_batch(patterns, graph_name)")
    print("  4. 预加载: preload_templates(['geography', 'company'])")
    print("  5. 清缓存: clear_cache()")