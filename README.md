# GPUStack Custom Provider

基于 [GPUStack](https://github.com/gpustack/gpustack) 的增强版本，扩展了自定义模型供应商的能力。

## 主要特性

- **双协议支持** — 自定义供应商同时支持 OpenAI 和 Anthropic API 格式，自动进行协议转换
- **双 Base URL** — 可分别为 OpenAI 和 Anthropic 格式配置独立的请求地址
- **双认证方式** — 可分别为两种格式配置不同的认证风格（Bearer / X-API-Key / Auto）
- **智能模型获取** — 自动适配版本化 API 路径（如智谱 `/v4`），兼容多种上游响应格式
- **非核心字段过滤** — 可选过滤 thinking、metadata 等字段，避免 Kimi 等非标准供应商报错

## 项目结构

```
gpustack/          # Python 后端（FastAPI + SQLModel）
gpustack-ui/       # React 前端（UmiJS Max + Ant Design 6）
```

## 快速开始

### 后端

```bash
cd gpustack
make install       # 安装开发工具
make deps          # 安装依赖
make test          # 运行测试
```

### 前端

```bash
cd gpustack-ui
pnpm install       # 安装依赖
pnpm build         # 构建生产版本
```

## 许可证

基于 GPUStack（Apache License 2.0）修改，详见 [LICENSE](gpustack/LICENSE) 和 [NOTICE](NOTICE)。
