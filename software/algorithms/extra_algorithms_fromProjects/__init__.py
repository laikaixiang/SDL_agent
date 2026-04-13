"""
项目扩展算法包 (software/algorithms/extra_algorithms_fromProjects/)
===================================================================

存放各项目中产生的自定义算法。

添加新算法的两种方式：

方式一（自动生成，推荐）：
    通过 Web 接口或 Python 调用 generate_algorithm()，用 LLM 自动生成代码文件：
    POST /api/software/generate_algorithm  Body: {"description": "需求描述..."}

方式二（手动编写）：
    参考 prompt_template.py 中的模板，手动创建符合接口的 .py 文件。

注意：
    - prompt_template.py 是生成器工具，SoftwareController 不会将其注册为算法
    - 所有需要注册的算法文件，类的 name 属性不能为空
"""
