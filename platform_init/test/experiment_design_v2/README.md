# 实验设计方案2测试

本目录包含实验设计方案2（JSON + 提示词）的测试代码。

## 测试内容

`test_experiment_design_v2.py` 验证以下功能：

1. **ExperimentDesignAgent初始化**
   - 从 `hardware/tools/REGISTRY.json` 加载硬件工具注册表
   - 从 `software/algorithms/` 动态加载软件算法
   - 内置6个辅助操作（WAIT, LOOP, GROUP, CONDITION, END, USER_INPUT）

2. **系统提示词生成**
   - 验证提示词包含所有硬件工具
   - 验证提示词包含所有软件算法
   - 验证提示词包含所有辅助操作

3. **JSON验证功能**
   - 测试有效的实验设计JSON
   - 测试无效的JSON格式（缺少必需字段）

4. **模拟实验生成**
   - 构造完整的实验设计JSON
   - 验证JSON格式正确性

## 运行测试

```bash
cd test/experiment_design_v2
python test_experiment_design_v2.py
```

## 预期输出

测试应该显示：
- 硬件工具数量: 5
- 软件算法数量: 4
- 辅助操作数量: 6
- 系统提示词长度: 2300+ 字符
- 所有工具和算法检查通过
- JSON验证功能正常

## 方案2说明

方案2使用JSON + 提示词的方式实现实验设计：
- **优势**: 不需要Function Calling支持，任何LLM都可使用
- **实现**: `core/field_inference.py:ExperimentDesignAgent`
- **注册表**: 
  - 硬件工具: `hardware/tools/REGISTRY.json`
  - 软件算法: 通过 `SoftwareController` 动态加载
  - 辅助操作: 内置在 `ExperimentDesignAgent` 中

## 相关文件

- `core/field_inference.py` - ExperimentDesignAgent实现
- `hardware/tools/REGISTRY.json` - 硬件工具注册表
- `software/algorithms/` - 软件算法目录
- `CLAUDE.md` - 完整文档
