import json
from typing import List, Dict, Optional, Union, Literal, Sequence
from openai import OpenAI


class SimpleLLMClient:
    """
    简化的LLM客户端，仅支持vLLM OpenAI兼容接口
    
    使用示例:
        client = SimpleLLMClient(
            provider="vllm",
            model="Qwen/Qwen2.5-7B-Instruct",
            base_url="http://<host>:8000/v1",
            api_key=""  # vLLM 通常不校验
        )

        response = client.generate(
            system="You are a helpful assistant",
            user_temp="What is the capital of France?"
        )
    """
    
    def __init__(
        self,
        provider: Literal["vllm"] = "vllm",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        enable_memory: bool = True,
        max_history: int = 10,
        persistent_system: str = "",
        persistent_context: str = ""
    ):
        """
        初始化LLM客户端
        
        参数:
            provider: 固定为 "vllm"（OpenAI 兼容接口）
            model: 模型名称（需与 vLLM 服务端可用的 model 名称一致）
            base_url: vLLM OpenAI 兼容接口地址，例如 "http://<host>:8000/v1"
            api_key: 可留空（多数 vLLM 不校验），但保持字段以兼容 OpenAI SDK
            temperature: 温度参数
            max_tokens: 最大生成token数
            enable_memory: 是否启用对话记忆
            max_history: 最大历史记录条数
            persistent_system: 持久化的系统提示（每次调用都会包含）
            persistent_context: 持久化的上下文信息（每次调用都会包含）
        """
        # 初始化 vLLM 的 OpenAI 兼容客户端
        if not base_url:
            raise ValueError("vLLM mode requires non-empty base_url (e.g., http://<host>:8000/v1)")
        self.provider = "vllm"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_memory = enable_memory
        self.max_history = max_history
        self.client = OpenAI(api_key=api_key or "", base_url=base_url)
        # 持久化内容
        self.persistent_system = persistent_system
        self.persistent_context = persistent_context
        # 对话历史记录
        self.conversation_history: List[Dict[str, str]] = []
    
    def generate(
        self,
        system: str = "",
        user: str = "",
        system_temp: str = "",
        user_temp: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reset_memory: bool = False
    ) -> str:
        """
        生成响应（主要接口）
        
        参数说明:
            system: 持久化系统提示（会保留在后续调用中）
            user: 持久化用户上下文（会保留在后续调用中）
            system_temp: 临时系统提示（仅本次调用使用）
            user_temp: 临时用户提示（仅本次调用使用）
            temperature: 临时温度设置（可选）
            max_tokens: 临时最大token设置（可选）
            reset_memory: 是否重置记忆
        
        使用场景:
            1. 固定任务指示 + 变化的问题:
               generate(system="You are an expert", user_temp="Question 1")
               generate(user_temp="Question 2")  # system会自动保留
            
            2. 固定上下文 + 变化的指令:
               generate(user="Schema: ...", system_temp="Generate query")
               generate(system_temp="Validate query")  # user上下文保留
            
            3. 完全临时调用:
               generate(system_temp="Task", user_temp="Question")
            
        返回:
            模型生成的文本响应
        """
        if reset_memory:
            self.reset_memory()
        
        # 更新持久化内容（如果提供了新的）
        if system:
            self.persistent_system = system
        if user:
            self.persistent_context = user
        
        # 构建消息列表
        messages = self._build_messages(system_temp, user_temp)
        
        # 调用模型
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )
            
            assistant_message = response.choices[0].message.content
            
            # 更新记忆（只保存临时的user提示和回复）
            if self.enable_memory and user_temp:
                self._update_memory(user_temp, assistant_message)
            
            return assistant_message
            
        except Exception as e:
            raise RuntimeError(f"LLM调用失败: {str(e)}")
    
    def _build_messages(self, system_temp: str, user_temp: str) -> List[Dict[str, str]]:
        """
        构建消息列表，组合顺序:
        1. persistent_system (持久化系统提示)
        2. system_temp (临时系统提示)
        3. persistent_context (持久化用户上下文)
        4. conversation_history (对话历史)
        5. user_temp (临时用户提示)
        """
        messages = []
        
        # 1. 组合系统消息
        system_parts = []
        if self.persistent_system:
            system_parts.append(self.persistent_system)
        if system_temp:
            system_parts.append(system_temp)
        
        if system_parts:
            combined_system = "\n\n".join(system_parts)
            messages.append({"role": "system", "content": combined_system})
        
        # 2. 添加持久化用户上下文
        if self.persistent_context:
            messages.append({"role": "user", "content": self.persistent_context})
            # 添加一个占位回复，表示已知晓上下文
            messages.append({"role": "assistant", "content": "Understood. I'll keep this context in mind."})
        
        # 3. 添加历史记录
        if self.enable_memory and self.conversation_history:
            messages.extend(self.conversation_history)
        
        # 4. 添加当前临时用户消息
        if user_temp:
            messages.append({"role": "user", "content": user_temp})
        
        return messages
    
    def _update_memory(self, user_msg: str, assistant_msg: str):
        """更新对话记忆"""
        self.conversation_history.append({"role": "user", "content": user_msg})
        self.conversation_history.append({"role": "assistant", "content": assistant_msg})
        
        # 限制历史记录长度（保留最近的对话）
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-(self.max_history * 2):]
    
    def reset_memory(self):
        """重置对话记忆"""
        self.conversation_history = []
    
    def reset_persistent(self):
        """重置所有持久化内容（system和context）"""
        self.persistent_system = ""
        self.persistent_context = ""
    
    def reset_all(self):
        """重置对话记忆和持久化内容"""
        self.reset_memory()
        self.reset_persistent()
    
    def update_persistent_system(self, system: str):
        """更新持久化系统提示"""
        self.persistent_system = system
    
    def update_persistent_context(self, context: str):
        """更新持久化用户上下文"""
        self.persistent_context = context
    
    def get_persistent_info(self) -> Dict[str, str]:
        """获取当前持久化内容"""
        return {
            "persistent_system": self.persistent_system,
            "persistent_context": self.persistent_context,
            "system_length": len(self.persistent_system),
            "context_length": len(self.persistent_context)
        }
    
    def get_memory(self) -> List[Dict[str, str]]:
        """获取当前对话记忆"""
        return self.conversation_history.copy()
    
    def set_memory(self, history: List[Dict[str, str]]):
        """手动设置对话记忆"""
        self.conversation_history = history
    
    def generate_with_json(
        self,
        system: str = "",
        user: str = "",
        system_temp: str = "",
        user_temp: str = "",
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> Union[str, Dict]:
        """
        生成JSON格式响应（vLLM OpenAI兼容接口，不保证支持结构化输出）
        
        参数:
            system: 持久化系统提示
            user: 持久化用户上下文
            system_temp: 临时系统提示
            user_temp: 临时用户提示
            response_format: JSON schema（OpenAI格式）
            **kwargs: 其他参数
            
        返回:
            JSON格式的响应
        """
        # 更新持久化内容
        if system:
            self.persistent_system = system
        if user:
            self.persistent_context = user
        
        # 构建消息列表
        messages = self._build_messages(system_temp, user_temp)
        
        try:
            # vLLM OpenAI 兼容接口：不保证支持 response_format，统一走普通调用
            if response_format is not None:
                # 为保持兼容性：如果调用方传了 schema，则在提示中请求返回 JSON
                if messages and messages[-1]["role"] == "user":
                    messages[-1]["content"] += "\n\nPlease respond in valid JSON format."

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens)
            )

            content = response.choices[0].message.content
            
            # 尝试解析JSON
            try:
                json_response = json.loads(content)
                if self.enable_memory and user_temp:
                    self._update_memory(user_temp, content)
                return json_response
            except json.JSONDecodeError:
                # 如果无法解析，返回原始文本
                if self.enable_memory and user_temp:
                    self._update_memory(user_temp, content)
                return content
                
        except Exception as e:
            raise RuntimeError(f"JSON生成失败: {str(e)}")
    
    def batch_generate(
        self,
        system: str = "",
        user: str = "",
        user_temp_list: Optional[Sequence[str]] = None,
        system_temp: str = "",
        reset_memory_each: bool = True
    ) -> List[str]:
        """
        批量生成响应
        
        参数:
            system: 持久化系统提示（对所有请求相同）
            user: 持久化用户上下文（对所有请求相同）
            user_temp_list: 临时用户提示列表（每个请求不同）
            system_temp: 临时系统提示（对所有请求相同）
            reset_memory_each: 每次调用是否重置记忆
            
        返回:
            响应列表
        """
        if user_temp_list is None:
            user_temp_list = []
        
        # 设置持久化内容
        if system:
            self.persistent_system = system
        if user:
            self.persistent_context = user
        
        responses = []
        
        for user_temp in user_temp_list:
            if reset_memory_each:
                self.reset_memory()
            
            response = self.generate(system_temp=system_temp, user_temp=user_temp)
            responses.append(response)
        
        return responses
    
    def __repr__(self):
        return (f"SimpleLLMClient(provider=vllm, "
                f"model={self.model}, "
                f"memory_enabled={self.enable_memory}, "
                f"history_length={len(self.conversation_history)})")


# ============= 使用示例 =============

if __name__ == "__main__":
    # 示例1: 持久化系统提示 + 临时问题（最常见场景）
    print("=== 示例1: 持久化系统提示 ===")
    client = SimpleLLMClient(
        provider="vllm",
        model="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:8000/v1",
        api_key="",
        temperature=0.7,
        enable_memory=True
    )
    
    # 设置持久化的系统提示（只需要设置一次）
    response1 = client.generate(
        system="You are a Cypher query expert specializing in Neo4j graph databases.",
        user_temp="What is a MATCH clause?"
    )
    print(f"Response 1: {response1}\n")
    
    # 后续调用时system会自动保留
    response2 = client.generate(
        user_temp="Can you give me an example?"
    )
    print(f"Response 2: {response2}\n")
    
    response3 = client.generate(
        user_temp="What about RETURN clause?"
    )
    print(f"Response 3: {response3}\n")
    
    
    # 示例2: 持久化上下文 + 临时任务（适合固定schema的查询生成）
    print("\n=== 示例2: 持久化上下文 ===")
    client.reset_all()  # 重置之前的内容
    
    schema = """
    Graph Schema:
    - Node: Person (name, age, city)
    - Node: Company (name, industry)
    - Relationship: WORKS_AT (Person -> Company)
    - Relationship: LIVES_IN (Person -> City)
    """
    
    # 设置持久化的schema上下文
    response1 = client.generate(
        user=schema,  # 持久化的上下文
        system_temp="Generate a Cypher query:",  # 临时任务
        user_temp="Find all people working at tech companies"
    )
    print(f"Query 1: {response1}\n")
    
    # schema会自动保留，不需要重复提供
    response2 = client.generate(
        system_temp="Generate a Cypher query:",
        user_temp="Find people living in New York"
    )
    print(f"Query 2: {response2}\n")
    
    # 可以更换任务指令，schema依然保留
    response3 = client.generate(
        system_temp="Explain this query pattern:",
        user_temp="MATCH (p:Person)-[:WORKS_AT]->(c:Company)"
    )
    print(f"Explanation: {response3}\n")
    
    
    # 示例3: 同时使用持久化system和user（复杂场景）
    print("\n=== 示例3: 双重持久化 ===")
    client.reset_all()
    
    rules = """
    Query Generation Rules:
    1. Always use WITH DISTINCT for deduplication
    2. Add ORDER BY for sorted results
    3. Use LIMIT for top-K queries
    """
    
    response = client.generate(
        system="You are a Cypher query generator.",  # 持久化系统
        user=f"{schema}\n\n{rules}",  # 持久化上下文
        user_temp="Generate query: Find top 5 oldest people in each company"
    )
    print(f"Generated Query: {response}\n")
    
    # 两者都会保留
    response2 = client.generate(
        user_temp="Generate query: Find companies with more than 10 employees"
    )
    print(f"Generated Query 2: {response2}\n")
    
    
    # 示例4: 查看和管理持久化内容
    print("\n=== 示例4: 持久化内容管理 ===")
    
    # 查看当前持久化内容
    info = client.get_persistent_info()
    print(f"当前持久化信息: {info}\n")
    
    # 单独更新system
    client.update_persistent_system("You are now a SPARQL expert.")
    
    # 单独更新context
    client.update_persistent_context("RDF Schema: ...")
    
    # 查看更新后的信息
    info = client.get_persistent_info()
    print(f"更新后的持久化信息: {info}\n")
    
    # 只重置对话历史，保留持久化内容
    client.reset_memory()
    
    # 重置所有（包括持久化内容）
    client.reset_all()
    
    
    # 示例5: 初始化时设置持久化内容
    print("\n=== 示例5: 初始化时设置 ===")
    client_with_persistent = SimpleLLMClient(
        provider="vllm",
        model="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:8000/v1",
        api_key="",
        persistent_system="You are a helpful database assistant.",
        persistent_context=schema,
        enable_memory=True
    )
    
    # 直接开始提问，system和context已经设置好
    response = client_with_persistent.generate(
        user_temp="Show me all nodes"
    )
    print(f"Response: {response}\n")
    
    
    # 示例6: 批量生成（共享持久化内容）
    print("\n=== 示例6: 批量生成 ===")
    client.reset_all()
    
    questions = [
        "Find all Person nodes",
        "Find all Company nodes",
        "Count WORKS_AT relationships"
    ]
    
    batch_responses = client.batch_generate(
        system="You are a Cypher query generator.",
        user=schema,
        user_temp_list=questions,
        system_temp="Generate a concise Cypher query:",
        reset_memory_each=True
    )
    
    for q, a in zip(questions, batch_responses):
        print(f"Q: {q}")
        print(f"A: {a}\n")
    
    
    # 示例7: 完全临时调用（不使用持久化）
    print("\n=== 示例7: 完全临时调用 ===")
    client.reset_all()
    
    response = client.generate(
        system_temp="You are a concise assistant.",
        user_temp="What is 2+2?"
    )
    print(f"Response: {response}\n")
    
    # 检查持久化内容（应该为空）
    info = client.get_persistent_info()
    print(f"持久化内容: {info}")