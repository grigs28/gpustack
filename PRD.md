# GPUStack 模型供应商管理增强 - 产品需求文档 (PRD)

## 1. 项目背景

### 1.1 现状
GPUStack 当前已支持 30+ 预设模型供应商（如 OpenAI、Claude、DeepSeek 等），通过 Higress AI-Proxy WASM 插件进行请求路由和协议转换。但存在以下限制：
- 仅支持预设供应商类型，无法添加自定义供应商
- 不支持 Anthropic 与 OpenAI 协议的双向自动转换
- 供应商仅支持单一后端地址，无负载均衡/故障转移能力

### 1.2 目标
参考 CC-Proxy 的单端口双格式架构和 Higress AI-Proxy 的协议转换能力，增强 GPUStack 的供应商管理模块，实现：
1. **自定义供应商** - 支持用户添加非预设供应商
2. **格式转换引擎** - Anthropic ↔ OpenAI 双向转换（请求+响应+流式）
3. **双 IP/多后端支持** - 供应商可配置多个后端地址

---

## 2. 功能需求

### 2.1 自定义供应商 (P0)
**需求描述**：允许用户添加非预设供应商，通过配置定义其行为。

**配置字段**：
```yaml
type: "custom"
name: "my-custom-provider"
customBaseUrl: "https://api.custom.com/v1"  # 支持多 URL 列表
supported_formats: ["openai", "anthropic"]   # 声明支持的协议格式
auth_style: "bearer"                         # 认证方式: auto/bearer/x-api-key
auth_header: "Authorization"                 # 自定义认证头
auth_prefix: "Bearer "                       # 认证前缀
strip_fields: true                           # 是否过滤非标准字段
```

**验收标准**：
- [ ] 前端表单支持选择 "custom" 供应商类型
- [ ] 后端 API 支持自定义供应商的 CRUD 操作
- [ ] 自定义供应商配置持久化到数据库

### 2.2 格式转换引擎 (P0)
**需求描述**：实现 Anthropic 与 OpenAI 协议的双向自动转换。

**转换策略优先级**：
1. 如果 provider.supported_formats 包含请求格式 → 直通
2. 如果 provider 明确声明支持目标格式 → 转换
3. 如果 provider 类型为 "custom" → 使用自定义映射
4. 默认尝试自动检测和转换

**自动检测规则**：
| 请求路径 | 检测格式 |
|---------|---------|
| `/v1/messages` | anthropic |
| `/v1/chat/completions` | openai |

**字段映射表**：

**Claude → OpenAI 请求转换**：
| Claude 字段 | OpenAI 字段 | 说明 |
|------------|------------|------|
| model | model | 直接映射 |
| system | messages[0] (role: system) | 系统提示词转为消息 |
| messages | messages | 消息数组拼接（content 数组→string） |
| max_tokens | max_tokens | 直接映射 |
| stop_sequences | stop | 字段名转换 |
| temperature | temperature | 直接映射 |
| top_p | top_p | 直接映射 |
| stream | stream | 直接映射 |
| tools | tools | 工具定义转换（input_schema→parameters） |
| tool_choice | tool_choice | 工具选择转换 |
| thinking.budget_tokens | reasoning_max_tokens | 推理配置转换 |

**流式响应转换**：
- SSE 流式事件逐事件转换
- OpenAI 的 `choices[0].delta.content` → Claude 的 `content_block_delta`
- Finish reason 映射：`stop`→`end_turn`, `length`→`max_tokens`, `tool_calls`→`tool_use`

**验收标准**：
- [ ] 非流式请求转换（双向）
- [ ] 流式响应转换（双向）
- [ ] 工具调用转换（双向）
- [ ] 多模态内容转换（图片 base64）
- [ ] 推理字段转换（thinking/reasoning_content）

### 2.3 双 IP/多后端支持 (P1)
**需求描述**：供应商可配置多个后端地址，支持负载均衡和故障转移。

**配置结构**：
```python
class Provider:
    name: str
    type: str                    # 预设类型或 "custom"
    api_keys: List[str]          # 支持多 key 轮询
    base_urls: List[str]         # 双 IP 或多 IP 支持
    supported_formats: List[str] # ["openai", "anthropic"]
    failover_strategy: str       # round_robin, random, priority
    health_check: bool = True
```

**验收标准**：
- [ ] 支持配置多个 base_url
- [ ] 支持轮询负载均衡策略
- [ ] 后端健康检查机制
- [ ] 故障自动转移

---

## 3. 技术方案

### 3.1 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                        GPUStack Gateway                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │  OpenAI API │    │ Anthropic   │    │   Format Converter  │ │
│  │  /v1/chat/* │    │ /v1/messages│    │  (New Module)       │ │
│  └──────┬──────┘    └──────┬──────┘    └─────────────────────┘ │
│         │                  │                  ▲                │
│         └──────────────────┼──────────────────┘                │
│                            │                                    │
│                   ┌────────┴────────┐                          │
│                   │  Router Logic   │                          │
│                   │  (Enhanced)     │                          │
│                   └────────┬────────┘                          │
│                            │                                    │
│         ┌──────────────────┼──────────────────┐                │
│         ▼                  ▼                  ▼                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   OpenAI    │    │  Anthropic  │    │   Custom    │         │
│  │  Provider   │    │  Provider   │    │  Provider   │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 核心模块

#### 3.2.1 格式转换模块 (gpustack/converter/)
新建模块，融合 CC-Proxy 和 Higress 的实现优势：

```
gpustack/converter/
├── __init__.py
├── base.py              # 基础转换器接口
├── claude_to_openai.py  # Claude → OpenAI 转换
├── openai_to_claude.py  # OpenAI → Claude 转换
├── streaming.py         # 流式响应转换
├── tools.py             # 工具调用转换
└── multimodal.py        # 多模态内容处理
```

**关键类设计**：
```python
class FormatConverter:
    """格式转换器主类"""
    
    @staticmethod
    def detect_format(request_path: str, body: dict) -> str:
        """自动检测请求协议"""
        
    @staticmethod
    def convert_request(body: dict, source: str, target: str) -> dict:
        """请求格式转换"""
        
    @staticmethod
    def convert_response(response: dict, source: str, target: str) -> dict:
        """响应格式转换"""
        
    @staticmethod
    def convert_streaming_chunk(chunk: dict, source: str, target: str) -> dict:
        """流式响应逐块转换"""
```

#### 3.2.2 数据模型扩展 (gpustack/schemas/model_provider.py)

**新增枚举**：
```python
class ModelProviderTypeEnum(str, Enum):
    # ... 现有类型 ...
    CUSTOM = "custom"  # 新增自定义类型

class AuthStyleEnum(str, Enum):
    AUTO = "auto"
    BEARER = "bearer"
    X_API_KEY = "x-api-key"
```

**新增配置类**：
```python
class CustomProviderConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.CUSTOM]
    customBaseUrl: Union[str, List[str]]  # 支持单 URL 或列表
    supported_formats: List[str] = ["openai"]  # ["openai", "anthropic"]
    auth_style: AuthStyleEnum = AuthStyleEnum.AUTO
    auth_header: Optional[str] = "Authorization"
    auth_prefix: Optional[str] = "Bearer "
    strip_fields: bool = True
    field_mapping: Optional[Dict[str, str]] = None  # 自定义字段映射
```

#### 3.2.3 路由逻辑增强 (gpustack/gateway/)

**智能路由逻辑**：
```python
class SmartRouter:
    """基于 supported_formats 的智能路由器"""
    
    def route_request(self, request: Request, provider: Provider) -> RouteDecision:
        request_format = self.detect_format(request)
        
        # 策略1: 如果供应商支持请求格式，直通
        if request_format in provider.supported_formats:
            return RouteDecision(target_format=request_format, convert=False)
        
        # 策略2: 需要转换到目标格式
        for target_format in provider.supported_formats:
            return RouteDecision(target_format=target_format, convert=True)
        
        # 策略3: 自定义供应商使用默认转换
        if provider.type == "custom":
            return RouteDecision(target_format="openai", convert=True)
```

#### 3.2.4 认证适配器

```python
class AuthAdapter:
    """根据目标供应商自动选择认证方式"""
    
    styles = {
        "openai": {"header": "Authorization", "prefix": "Bearer "},
        "anthropic_auto": {
            "header": "Authorization", 
            "prefix": "Bearer ",
            "extra_header": "x-api-key", 
            "extra_prefix": ""
        },
        "anthropic_bearer": {"header": "Authorization", "prefix": "Bearer "},
        "anthropic_x_api_key": {"header": "x-api-key", "prefix": ""},
    }
```

---

## 4. 修改清单

### 4.1 后端修改

| 文件路径 | 修改类型 | 说明 |
|---------|---------|------|
| `gpustack/schemas/model_provider.py` | 修改 | 添加 CUSTOM 类型、CustomProviderConfig 类、AuthStyleEnum |
| `gpustack/gateway/ai_proxy_types.py` | 修改 | 扩展 AIProxyDefaultConfig 支持自定义配置 |
| `gpustack/routes/model_provider.py` | 修改 | 增强验证逻辑，支持自定义供应商 |
| `gpustack/routes/openai.py` | 修改 | 集成格式转换中间件 |
| `gpustack/routes/routes.py` | 修改 | 如有需要，添加新的路由 |
| `gpustack/converter/__init__.py` | 新建 | 转换器模块初始化 |
| `gpustack/converter/base.py` | 新建 | 基础转换器接口 |
| `gpustack/converter/claude_to_openai.py` | 新建 | Claude → OpenAI 转换实现 |
| `gpustack/converter/openai_to_claude.py` | 新建 | OpenAI → Claude 转换实现 |
| `gpustack/converter/streaming.py` | 新建 | 流式响应转换 |
| `tests/test_converter.py` | 新建 | 转换器单元测试 |

### 4.2 前端修改

| 文件路径 | 修改类型 | 说明 |
|---------|---------|------|
| `src/pages/maas-provider/config/providers.ts` | 修改 | 添加 CUSTOM 到 ProviderEnum |
| `src/pages/maas-provider/config/types.ts` | 修改 | 扩展 FormData 接口支持自定义字段 |
| `src/pages/maas-provider/forms/index.tsx` | 修改 | 支持 custom 类型表单渲染 |
| `src/pages/maas-provider/forms/basic.tsx` | 修改 | 添加自定义供应商基础配置表单 |
| `src/locales/*/provider.ts` | 修改 | 添加自定义供应商国际化文案 |

---

## 5. 构建与部署流程

### 5.1 开发环境

**后端开发**：
```bash
cd /opt/gpustack/gpustack
uv run gpustack start
```

**前端开发**：
```bash
cd /opt/gpustack/gpustack-ui
pnpm install
pnpm dev
```

### 5.2 构建部署

**完整构建流程**：
```bash
# 1. 前端构建
cd /opt/gpustack/gpustack-ui
pnpm build

# 2. 复制构建产物到后端
cp -r dist/* /opt/gpustack/gpustack/gpustack/ui/

# 3. 构建 wheel 包
cd /opt/gpustack/gpustack
python -m build --wheel
mv dist/gpustack-*.whl /opt/gpustack.glm/gpustack-custom.whl

# 4. 构建 Docker 镜像
cd /opt/gpustack.glm
# 使用 pack/Dockerfile.custom 构建
docker build -f pack/Dockerfile.custom -t gpustack/gpustack:custom .
```

**Dockerfile.custom 说明**：
```dockerfile
# Lightweight rebuild: install modified gpustack on top of the existing image
FROM gpustack/gpustack:latest

# Copy the built wheel (placed outside dist/ to avoid .dockerignore)
COPY gpustack-custom.whl /tmp/

# Backup UI files, reinstall gpustack, then restore UI
RUN cp -r /usr/local/lib/python3.11/dist-packages/gpustack/ui /tmp/gpustack-ui-backup \
    && mv /tmp/gpustack-custom.whl /tmp/gpustack-0.0.0-py3-none-any.whl \
    && uv pip install --system --no-build-isolation --force-reinstall /tmp/gpustack-0.0.0-py3-none-any.whl \
    && cp -r /tmp/gpustack-ui-backup /usr/local/lib/python3.11/dist-packages/gpustack/ui \
    && rm -rf /tmp/gpustack-ui-backup /tmp/gpustack-0.0.0-py3-none-any.whl
```

---

## 6. 验收标准

### 6.1 功能验收

| 功能 | 验收标准 | 优先级 |
|-----|---------|-------|
| 自定义供应商创建 | 可通过 UI 创建 type="custom" 的供应商 | P0 |
| 自定义供应商编辑 | 可修改自定义供应商配置 | P0 |
| 自定义供应商删除 | 可删除自定义供应商 | P0 |
| Anthropic→OpenAI 请求转换 | /v1/messages 请求正确转换为 /v1/chat/completions | P0 |
| OpenAI→Anthropic 请求转换 | /v1/chat/completions 请求正确转换为 /v1/messages | P0 |
| 非流式响应转换 | 响应格式正确转换 | P0 |
| 流式响应转换 | SSE 流逐块正确转换 | P0 |
| 工具调用转换 | tools/tool_calls 字段正确转换 | P0 |
| 多模态内容转换 | 图片 base64 内容正确转换 | P1 |
| 多后端配置 | 支持配置多个 base_url | P1 |
| 负载均衡 | 多后端轮询策略正常工作 | P1 |
| 健康检查 | 后端健康检查机制正常工作 | P1 |

### 6.2 性能验收

- 格式转换延迟 < 10ms（单次请求）
- 流式转换无感知延迟
- 支持并发请求处理

### 6.3 测试覆盖

- 单元测试覆盖率 > 80%
- 集成测试覆盖主要转换场景
- 边界情况测试（空内容、错误响应等）

---

## 7. 参考资源

### 7.1 代码参考

| 资源 | 路径/URL | 用途 |
|-----|---------|------|
| CC-Proxy | `/opt/cc-proxy/cc_proxy/converter.py` | 格式转换核心逻辑参考 |
| Higress AI-Proxy | https://github.com/alibaba/higress/tree/main/plugins/wasm-go/extensions/ai-proxy | 协议转换实现参考 |
| GPUStack 后端 | `/opt/gpustack/gpustack/gpustack` | 现有后端代码 |
| GPUStack 前端 | `/opt/gpustack/gpustack-ui/src` | 现有前端代码 |

### 7.2 API 文档

- [Anthropic Messages API](https://docs.anthropic.com/en/api/messages)
- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)
- [Higress AI-Proxy 文档](https://higress.cn/docs/latest/plugins/ai/api-provider/ai-proxy/)

---

## 8. 风险与问题

| 风险 | 影响 | 缓解措施 |
|-----|------|---------|
| 格式转换复杂度 | 高 | 复用 CC-Proxy 已验证的逻辑，逐步迭代 |
| 流式转换性能 | 中 | 使用生成器模式，避免内存拷贝 |
| 多后端状态同步 | 中 | 先实现简单轮询，再扩展健康检查 |
| 与 Higress 集成冲突 | 中 | 保持与现有 Higress 配置的兼容性 |

---

## 9. 附录

### 9.1 术语表

| 术语 | 说明 |
|-----|------|
| CC-Proxy | 参考项目，单端口双格式代理实现 |
| Higress | 阿里云开源的 AI 网关，支持 WASM 插件 |
| AI-Proxy | Higress 的 AI 供应商代理插件 |
| SSE | Server-Sent Events，流式响应协议 |

### 9.2 变更日志

| 日期 | 版本 | 变更内容 |
|-----|------|---------|
| 2026-04-13 | v1.0 | 初始版本 |
