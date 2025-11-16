# API Keys 获取详细指南

本文档提供三个数据源（CrossRef、OpenReview、Unpaywall）的详细配置步骤。

## 重要说明

✅ **好消息：这三个源都不需要传统的API key！**

它们只需要：
- **CrossRef**: 在User-Agent header中包含邮箱
- **OpenReview**: 在User-Agent header中包含邮箱  
- **Unpaywall**: 使用email参数（但Unpaywall不支持搜索API）

---

## 1. CrossRef API

### 获取方式：
✅ **完全免费，无需注册，无需API key！**

### 配置步骤：

#### 方法1：使用环境变量（推荐）

```bash
export CROSSREF_EMAIL="your.email@example.com"
```

#### 方法2：在代码中直接设置

代码已经自动从环境变量读取，如果没有设置，会使用默认值 `litreview@example.com`

### 使用说明：
- CrossRef API：https://api.crossref.org/works
- 官方文档：https://www.crossref.org/services/metadata-delivery/rest-api/
- 使用限制：无明确限制，建议设置User-Agent包含邮箱以获得更好服务

### 验证配置：
运行代码时，如果看到"从CrossRef获得 X 篇唯一论文"，说明配置成功。

---

## 2. OpenReview API

### 获取方式：
✅ **免费，无需API key，但建议注册账户**

### 配置步骤：

#### 步骤1：注册OpenReview账户（可选但推荐）
1. 访问：https://openreview.net/
2. 点击右上角 **"Sign Up"** 或 **"注册"**
3. 填写注册信息（邮箱、用户名、密码等）
4. 验证邮箱（如果需要）

#### 步骤2：设置邮箱（用于User-Agent）

```bash
export OPENREVIEW_EMAIL="your.email@example.com"
```

### 使用说明：
- OpenReview API：https://api2.openreview.net
- API文档：https://docs.openreview.net/getting-started/using-the-api
- 限制：无明确限制，但建议设置User-Agent包含邮箱

### 验证配置：
运行代码时，如果看到"从OpenReview获得 X 篇唯一论文"，说明配置成功。

### 注意事项：
- OpenReview不支持文本搜索，代码会从特定会议（ICLR、NeurIPS、ICML等）获取论文
- 然后通过关键词匹配筛选相关论文
- 因此OpenReview的结果可能不如其他源多，但质量较高

---

## 3. Unpaywall API

### 获取方式：
⚠️ **需要注册邮箱，但不需要API key**

### 重要说明：
❌ **Unpaywall不支持搜索API！**
- Unpaywall只能通过DOI查询单个论文的open access信息
- 不能用于初始的论文搜索
- 代码中会提示Unpaywall不能用于搜索

### 配置步骤（如果将来需要）：

#### 步骤1：访问Unpaywall官网
1. 访问：https://unpaywall.org/products/api
2. 查看API文档

#### 步骤2：注册邮箱（用于API请求）
- Unpaywall使用email参数而不是API key
- 在API请求中传递 `email` 参数

### 使用方式（通过DOI查询）：
```python
url = f"https://api.unpaywall.org/v2/{doi}?email=your.email@example.com"
```

### 限制：
- 免费版：每天100,000次请求
- 需要邮箱注册（建议使用机构邮箱）

### 当前状态：
代码中已实现Unpaywall检索框架，但由于Unpaywall不支持搜索，会跳过并提示。

---

## 快速配置指南

### 步骤1：设置邮箱环境变量

在shell中运行：
```bash
# CrossRef邮箱（可选，但推荐）
export CROSSREF_EMAIL="your.email@example.com"

# OpenReview邮箱（可选，但推荐）
export OPENREVIEW_EMAIL="your.email@example.com"

# 注意：Unpaywall暂时不需要，因为它不支持搜索API
```

### 步骤2：验证配置

运行测试：
```bash
cd /data6/zhengqirui/litllms-for-literature-review-tmlr/retrieval
python src/integrated_workflow.py \
  --topic "test topic" \
  --n_candidates 10 \
  --n_papers_for_generation 10 \
  --search_sources arxiv openalex crossref openreview
```

如果看到以下输出，说明配置成功：
```
[3/5] 从CrossRef检索...
从CrossRef获得 X 篇唯一论文

[4/5] 从OpenReview检索...
从OpenReview获得 X 篇唯一论文
```

---

## 完整配置示例

### 方法1：临时设置（当前shell会话）

```bash
export CROSSREF_EMAIL="your.email@example.com"
export OPENREVIEW_EMAIL="your.email@example.com"

# 然后运行
python src/integrated_workflow.py --topic "..." ...
```

### 方法2：永久设置（添加到 ~/.bashrc 或 ~/.bash_profile）

```bash
# 编辑配置文件
nano ~/.bashrc

# 添加以下行
export CROSSREF_EMAIL="your.email@example.com"
export OPENREVIEW_EMAIL="your.email@example.com"

# 保存并重新加载
source ~/.bashrc
```

### 方法3：创建 .env 文件（需要安装python-dotenv）

```bash
# 在项目根目录创建 .env 文件
cat > .env << EOF
CROSSREF_EMAIL=your.email@example.com
OPENREVIEW_EMAIL=your.email@example.com
EOF
```

然后在代码开头添加：
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 总结

| 数据源 | 是否需要API key | 需要什么 | 状态 |
|--------|----------------|----------|------|
| CrossRef | ❌ 否 | 邮箱（用于User-Agent） | ✅ 已实现 |
| OpenReview | ❌ 否 | 邮箱（用于User-Agent） | ✅ 已实现 |
| Unpaywall | ❌ 否 | 邮箱（但不支持搜索API） | ⚠️ 不支持搜索 |
| arXiv | ❌ 否 | 无需配置 | ✅ 已实现 |
| OpenAlex | ❌ 否 | 无需配置 | ✅ 已实现 |

---

## 常见问题

### Q1: 我没有设置邮箱，能使用吗？
A: 可以，代码会使用默认邮箱 `litreview@example.com`，但建议使用自己的邮箱以获得更好的服务。

### Q2: Unpaywall为什么不能用于搜索？
A: Unpaywall的设计目标不是论文搜索，而是通过DOI查询论文的open access信息。如果需要搜索功能，请使用其他源。

### Q3: 如何知道配置是否成功？
A: 运行代码时，查看控制台输出。如果看到"从CrossRef获得 X 篇唯一论文"和"从OpenReview获得 X 篇唯一论文"，说明配置成功。

### Q4: 可以只使用部分数据源吗？
A: 可以！使用 `--search_sources` 参数指定：
```bash
# 只使用arXiv和OpenAlex
--search_sources arxiv openalex

# 使用所有可用的源
--search_sources arxiv openalex crossref openreview
```

---

## 下一步

1. ✅ 设置邮箱环境变量（可选）
2. ✅ 运行测试，验证所有源都能正常工作
3. ✅ 开始使用完整的检索-生成工作流

**无需额外的API key获取步骤！代码已经配置好，可以直接使用！**
