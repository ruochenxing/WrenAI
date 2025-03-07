import logging
import sys
from typing import Any

import orjson
from hamilton import base
from hamilton.async_driver import AsyncDriver
from haystack.components.builders.prompt_builder import PromptBuilder
from langfuse.decorators import observe
from pydantic import BaseModel

from src.core.pipeline import BasicPipeline
from src.core.provider import LLMProvider

logger = logging.getLogger("wren-ai-service")


# 数据模型处理函数
@observe(capture_input=False)
def picked_models(mdl: dict, selected_models: list[str]) -> list[dict]:
    """
    处理选中的数据模型，提取相关信息并格式化
    :param mdl: 包含所有模型信息的字典
    :param selected_models: 被选中的模型名称列表
    :return: 格式化后的模型列表
    """

    def relation_filter(column: dict) -> bool:
        # 过滤掉关系型列
        return "relationship" not in column

    def column_formatter(columns: list[dict]) -> list[dict]:
        # 格式化列信息，提取名称、类型和描述
        return [
            {
                "name": column["name"],
                "type": column["type"],
                "properties": {
                    "description": column["properties"].get("description", ""),
                },
            }
            for column in columns
            if relation_filter(column)
        ]

    def extract(model: dict) -> dict:
        # 提取模型的核心信息
        return {
            "name": model["name"],
            "columns": column_formatter(model["columns"]),
            "properties": {
                "description": model["properties"].get("description", ""),
            },
        }

    return [
        extract(model)
        for model in mdl.get("models", [])
        if model.get("name", "") in selected_models
    ]


# 提示词生成函数
@observe(capture_input=False)
def prompt(
    picked_models: list[dict],
    user_prompt: str,
    prompt_builder: PromptBuilder,
    language: str,
) -> dict:
    """
    构建用于生成语义描述的提示词
    :param picked_models: 处理后的模型列表
    :param user_prompt: 用户输入的提示
    :param prompt_builder: 提示词构建器
    :param language: 目标语言
    :return: 构建好的提示词
    """
    return prompt_builder.run(
        picked_models=picked_models,
        user_prompt=user_prompt,
        language=language,
    )


# LLM生成函数
@observe(as_type="generation", capture_input=False)
async def generate(prompt: dict, generator: Any) -> dict:
    """
    使用LLM生成语义描述
    :param prompt: 提示词
    :param generator: LLM生成器
    :return: 生成的结果
    """
    return await generator(prompt=prompt.get("prompt"))


# 响应标准化函数
@observe(capture_input=False)
def normalize(generate: dict) -> dict:
    """
    标准化LLM的输出结果
    :param generate: LLM生成的原始结果
    :return: 标准化后的字典
    """

    def wrapper(text: str) -> str:
        # 清理和格式化文本
        text = text.replace("\n", " ")
        text = " ".join(text.split())
        try:
            text_dict = orjson.loads(text.strip())
            return text_dict
        except orjson.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            return {"models": []}

    reply = generate.get("replies")[0]
    normalized = wrapper(reply)
    return {model["name"]: model for model in normalized["models"]}


# 输出处理函数
@observe(capture_input=False)
def output(normalize: dict, picked_models: list[dict]) -> dict:
    """
    处理最终输出，确保输出的列与原始模型匹配
    :param normalize: 标准化后的数据
    :param picked_models: 原始选中的模型
    :return: 最终处理后的输出
    """

    def _filter(enriched: list[dict], columns: list[dict]) -> list[dict]:
        valid_columns = [col["name"] for col in columns]
        return [col for col in enriched if col["name"] in valid_columns]

    models = {model["name"]: model for model in picked_models}
    return {
        name: {**data, "columns": _filter(data["columns"], models[name]["columns"])}
        for name, data in normalize.items()
        if name in models
    }


# 数据模型定义
class ModelProperties(BaseModel):
    """模型属性类，包含描述信息"""

    description: str


class ModelColumns(BaseModel):
    """模型列类，定义列的结构"""

    name: str
    properties: ModelProperties


class SemanticModel(BaseModel):
    """语义模型类，定义完整的模型结构"""

    name: str
    columns: list[ModelColumns]
    properties: ModelProperties


class SemanticResult(BaseModel):
    """语义结果类，包含多个模型的结果"""

    models: list[SemanticModel]


# LLM响应格式配置
SEMANTICS_DESCRIPTION_MODEL_KWARGS = {
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "semantic_description",
            "schema": SemanticResult.model_json_schema(),
        },
    }
}

# 系统提示词，指导LLM如何生成描述
system_prompt = """
I have a data model represented in JSON format, with the following structure:

```
[
    {'name': 'model', 'columns': [
            {'name': 'column_1', 'type': 'type', 'properties': {}
            },
            {'name': 'column_2', 'type': 'type', 'properties': {}
            },
            {'name': 'column_3', 'type': 'type', 'properties': {}
            }
        ], 'properties': {}
    }
]
```

Your task is to update this JSON structure by adding a `description` field inside both the `properties` attribute of each `column` and the `model` itself.
Each `description` should be derived from a user-provided input that explains the purpose or context of the `model` and its respective columns.
Follow these steps:
1. **For the `model`**: Prompt the user to provide a brief description of the model's overall purpose or its context. Insert this description in the `properties` field of the `model`.
2. **For each `column`**: Ask the user to describe each column's role or significance. Each column's description should be added under its respective `properties` field in the format: `'description': 'user-provided text'`.
3. Ensure that the output is a well-formatted JSON structure, preserving the input's original format and adding the appropriate `description` fields.

### Output Format:

```
{
    "models": [
        {
        "name": "model",
        "columns": [
            {
                "name": "column_1",
                "properties": {
                    "description": "<description for column_1>"
                }
            },
            {
                "name": "column_2",
                "properties": {
                    "description": "<description for column_1>"
                }
            },
            {
                "name": "column_3",
                "properties": {
                    "description": "<description for column_1>"
                }
            }
        ],
        "properties": {
                "description": "<description for model>"
            }
        }
    ]
}
```

Make sure that the descriptions are concise, informative, and contextually appropriate based on the input provided by the user.
"""

# 用户提示词模板
user_prompt_template = """
### Input:
User's prompt: {{ user_prompt }}
Picked models: {{ picked_models }}
Localization Language: {{ language }}

Please provide a brief description for the model and each column based on the user's prompt.
"""


class SemanticsDescription(BasicPipeline):
    """
    语义描述生成管道类
    用于根据用户输入为数据模型及其列生成语义化的描述
    继承自BasicPipeline基类
    """

    def __init__(self, llm_provider: LLMProvider, **_):
        """
        初始化语义描述生成管道
        :param llm_provider: LLM提供者，用于生成描述
        :param _: 其他参数
        """
        self._components = {
            "prompt_builder": PromptBuilder(template=user_prompt_template),
            "generator": llm_provider.get_generator(
                system_prompt=system_prompt,
                generation_kwargs=SEMANTICS_DESCRIPTION_MODEL_KWARGS,
            ),
        }
        self._final = "output"

        super().__init__(
            AsyncDriver({}, sys.modules[__name__], result_builder=base.DictResult())
        )

    @observe(name="Semantics Description Generation")
    async def run(
        self,
        user_prompt: str,
        selected_models: list[str],
        mdl: dict,
        language: str = "en",
    ) -> dict:
        """
        运行语义描述生成管道
        :param user_prompt: 用户输入的提示
        :param selected_models: 选中的模型列表
        :param mdl: 模型数据
        :param language: 目标语言，默认为英语
        :return: 生成的语义描述结果
        """
        logger.info("Semantics Description Generation pipeline is running...")
        return await self._pipe.execute(
            [self._final],
            inputs={
                "user_prompt": user_prompt,
                "selected_models": selected_models,
                "mdl": mdl,
                "language": language,
                **self._components,
            },
        )


# 主函数，用于测试
if __name__ == "__main__":
    from src.pipelines.common import dry_run_pipeline

    dry_run_pipeline(
        SemanticsDescription,
        "semantics_description",
        user_prompt="Track student enrollments, grades, and GPA calculations to monitor academic performance and identify areas for student support",
        selected_models=[],
        mdl={},
        language="en",
    )
