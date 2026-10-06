# Currency Analytics

[Русский](README.ru.md) · [English](README.md) · **中文** · [हिन्दी](README.hi.md) · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md)

研究型汇率分析应用，使用俄罗斯央行数据分析 USD/RUB、EUR/RUB、CNY/RUB、GBP/RUB。作为补充作品集项目，目前不扩展功能。

## 功能与技术

Python 3.10+、FastAPI、pandas、NumPy、LightGBM、XGBoost 和 scikit-learn；HTML/CSS/JS 与本地 Chart.js。没有训练模型时使用统计趋势回退。TF-IDF 检索知识库；代码进行数值计算，Ollama 生成开放式回答，sources 提供检索来源。Redis 不可用时回退至进程内缓存。SQLite/SQLAlchemy 记录 A/B 分配和预测比较。Prometheus、JSON 日志和可选 Grafana 用于监测。

## 评估边界

单元与 smoke 测试不能证明预测优于持久性基线（明天等于今天）、交易盈利或生产部署。代码包含 walk-forward 评估；可信比较需要按时间留出的数据、数据哈希、配置和原始结果。A/B 分配、MAE/MAPE 或 t-test 本身不能证明有效实验；必须考虑时间依赖与重复观测。区间并非可靠性的独立证明，不应据此作投资决策。

## 安装、启动与测试

先根据部署环境配置应用所需环境变量和自有 SECRET_KEY/JWT_SECRET_KEY；配置模板见仓库，勿将秘密提交到 Git。以下为示例，不是实际部署证据。CI 测试 Python 3.10/3.11；测试不依赖在线央行或 Ollama 服务。

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pytest tests/ -v
.venv/bin/python -m uvicorn src.main:app --host 127.0.0.1 --port 8002
```

```sh
docker compose up -d --build
curl http://localhost:8002/api/health
docker compose logs -f
```

```sh
.venv/bin/python train_models.py
.venv/bin/python scripts/update_ab_actual_rates.py
```

## 接口与示例

历史数据、预测、统计、助手和健康检查的路径如下。/api/refresh、/api/force-refresh 和 /api/cache/status 需要 X-Admin-Key。助手响应包含 sources。A/B 日志通过 /api/ab-test/* 访问；refresh-actuals 更新实际汇率后才能比较。

```text
GET  /api/data/data?period_days=30
GET  /api/forecast/forecast?days=7&currency=USD
GET  /api/stats/stats
GET  /api/health
GET  /api/ping
POST /api/rag/ask
GET  /api/ab-test/status
POST /api/ab-test/predict?currency=usd_rate&days=1
GET  /api/ab-test/stats?days=30
POST /api/ab-test/update-ratios?split_a=0.5
POST /api/ab-test/refresh-actuals
```

```sh
curl 'http://localhost:8002/api/forecast/forecast?days=7&currency=USD'
curl -X POST http://localhost:8002/api/rag/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the USD forecast for next week?"}'
```

## 运维与限制

Compose 发布主机端口 8002，默认并非仅 loopback；配置网络访问范围后再部署。/monitoring/dashboard 提供监测页面，/metrics 提供 Prometheus，/docs 和 /redoc 提供 API 文档。Ollama 和 MLflow 需单独运行；MLFLOW_TRACKING_URI 为空可关闭跟踪。768 MiB 容器限制、单 worker 和 4 GiB VDS 设计目标不是资源基准。monitoring profile 可选。Compose 的 root 用户覆盖、安全配置、依赖审查、备份和 HTTPS 仍需部署者验证。

## 语言与许可

这些是简明本地化指南；完整示例见英文或俄文 README。应用界面、助手能力和日志没有因此增加语言支持。许可采用仓库 LICENSE 的原文；预测仅供信息和教育用途。

[English](README.md) · [Русский](README.ru.md) · [LICENSE](LICENSE)
