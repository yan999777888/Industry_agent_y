# 当前开发重点

## RAG 链路短板判断

当前主要短板不是单纯“大模型接口不够优秀”，而是知识库组织、数据清洗、切分元数据和检索排序仍会影响上下文质量。若检索返回的 chunk 错域、偏安全警告、偏目录或缺少操作步骤，即使换成更强模型，也容易生成低质量回答。

## 数据清洗已完成

- 已扩展手册解析阶段的 OCR 粘连修复，覆盖 `tocleanany`、`sup ply`、`thepoweroutlet`、`beforesetting` 等常见英文手册噪声。
- 已扩展 `汇总英文` 的子领域标签，新增并强化 `lawn_mower`、`coffee_machine`、`fax`、`toothbrush`、`grill`、`earphone`、`television`、`washing_machine` 等领域识别；当前英文汇总 chunk 已全部完成领域标注。
- 已为 chunk 增加语义类型元数据：`procedure`、`safety_warning`、`troubleshooting`、`parts_list`、`specification`、`toc`、`general`。
- 已在检索排序中使用清洗元数据，针对操作类、故障类、安全类、部件类和规格类问题优先匹配对应语义 chunk。
- 已过滤空标题、孤立保修标题、残句等低价值碎片，减少无效召回。
- 已重建知识库索引，当前 `data/processed/kb/build_summary.json` 显示 `chunk_count=3800`、`fts5_available=true`。

## 下一步建议

- 用固定回归集和真实接口样例观察新增语义排序是否改善“错域”和“安全警告压过操作步骤”的问题。
- 若仍有明显错检，再继续针对 `汇总英文` 做更细的产品边界切分，而不是直接追公开题数据。
- 保持生成模型不变时，优先提高检索上下文质量；只有在上下文正确但回答表达仍弱时，再考虑更强 LLM API。
