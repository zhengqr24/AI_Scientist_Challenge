# API Keys 获取指南

本文档说明如何获取各个数据源的API key，以启用多源检索功能。

## 1. CrossRef API

### 获取方式：
- **免费**：不需要API key，可以直接使用REST API
- **官方文档**：https://www.crossref.org/services/metadata-delivery/rest-api/
- **使用限制**：需要设置User-Agent header（建议包含邮箱）

### 配置：
无需API key，但建议设置邮箱：
```python
headers = {
    "User-Agent": "YourApp/1.0 (mailto:your.email@example.com)"
}
```

## 2. OpenReview API

### 获取方式：
- **免费**：需要注册OpenReview账户
- **注册地址**：https://openreview.net/
- **API文档**：https://docs.openreview.net/
- **API key获取**：
  1. 登录OpenReview
  2. 访问个人设置页面
  3. 申请API access（需要说明用途）
  4. 获取API token

### 配置：
```bash
export OPENREVIEW_API_KEY="your-openreview-api-key"
```

或在代码中：
```python
OPENREVIEW_API_KEY = "your-openreview-api-key"
```

## 3. Unpaywall API

### 获取方式：
- **免费**：需要注册获取API key
- **注册地址**：https://unpaywall.org/products/api
- **限制**：免费版限制100,000 requests/day，需要邮箱注册

### 配置：
1. 访问 https://unpaywall.org/products/api
2. 使用邮箱注册
3. 获取API key（通常会在邮箱中收到）
4. 设置环境变量：
```bash
export UNPAYWALL_API_KEY="your-unpaywall-api-key"
```

或在代码中：
```python
UNPAYWALL_API_KEY = "your-unpaywall-api-key"
```

## 4. OpenAlex（当前已支持）

### 获取方式：
- **免费**：不需要API key，可以直接使用
- **官方文档**：https://docs.openalex.org/
- **使用限制**：建议设置User-Agent header包含邮箱

### 配置：
已在代码中自动配置，无需额外设置。

## 5. arXiv（当前已支持）

### 获取方式：
- **免费**：不需要API key
- **官方文档**：https://arxiv.org/help/api

### 配置：
已在代码中自动配置，无需额外设置。

## 环境变量配置示例

创建 `.env` 文件或在shell中设置：

```bash
# DeepSeek API (必需)
export DEEPSEEK_API_KEY="your-deepseek-api-key"

# OpenReview API (可选)
export OPENREVIEW_API_KEY="your-openreview-api-key"

# Unpaywall API (可选)
export UNPAYWALL_API_KEY="your-unpaywall-api-key"
```

## 使用优先级

1. **arXiv** - 已实现，无需配置
2. **OpenAlex** - 已实现，无需配置
3. **CrossRef** - 无需API key，但需要实现API调用代码
4. **Unpaywall** - 需要API key，需要实现API调用代码
5. **OpenReview** - 需要API key，需要实现API调用代码

## 注意事项

- CrossRef和Unpaywall的API调用需要实现相应的函数
- OpenReview可能需要特殊的认证方式
- 建议先从arXiv和OpenAlex开始，这两个源已完全实现
- 如果需要其他源的完整实现，需要添加对应的API调用代码

