import logging
import sys
from datetime import datetime
from typing import Any

import orjson
from hamilton import base
from hamilton.async_driver import AsyncDriver
from haystack.components.builders.prompt_builder import PromptBuilder
from langfuse.decorators import observe
from pydantic import BaseModel

from src.core.pipeline import BasicPipeline
from src.core.provider import LLMProvider

logger = logging.getLogger("pipeline.generation.question_recommendation")


## Start of Pipeline
@observe(capture_input=False)
def prompt(
    mdl: dict,
    previous_questions: list[str],
    language: str,
    current_date: str,
    max_questions: int,
    max_categories: int,
    prompt_builder: PromptBuilder,
) -> dict:
    """
    为LLM构建生成问题推荐的提示。

    如果提供了previous_questions，则会省略MDL，以便LLM能够专注于
    基于问题历史生成推荐。这有助于提供更多
    与上下文相关的问题，这些问题建立在之前的问题之上。

    参数:
        mdl: 包含数据模型规范的字典
        previous_questions: 要在其基础上构建的先前问题列表
        language: 生成问题的目标语言
        current_date: 上下文的当前日期字符串
        max_questions: 要生成的最大问题数
        max_categories: 要包含的最大类别数
        prompt_builder: 从模板构建提示的组件

    返回:
        包含构建的提示的字典
    """

    return prompt_builder.run(
        models=[] if previous_questions else mdl.get("models", []),
        previous_questions=previous_questions,
        language=language,
        current_date=current_date,
        max_questions=max_questions,
        max_categories=max_categories,
    )


@observe(capture_input=False, as_type="generation")
async def generate(prompt: dict, generator: Any) -> dict:
    """
    使用LLM生成问题推荐。

    参数:
        prompt: 包含要发送给LLM的提示的字典
        generator: LLM生成器函数/对象

    返回:
        包含LLM响应的字典
    """
    return await generator(prompt=prompt.get("prompt"))


@observe(capture_input=False)
def normalized(generate: dict) -> dict:
    """
    通过解析JSON输出来标准化LLM的响应。

    处理来自LLM的原始文本响应，清理它，并
    尝试将其解析为JSON以提取结构化的问题推荐。

    参数:
        generate: 包含LLM响应的字典

    返回:
        标准化的问题推荐列表
    """

    def wrapper(text: str) -> list:
        # 清理文本，移除换行符和多余的空格
        text = text.replace("\n", " ")
        text = " ".join(text.split())
        try:
            # 将文本解析为JSON
            text_list = orjson.loads(text.strip())
            return text_list
        except orjson.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            return []  # 如果JSON解析失败，返回空列表

    reply = generate.get("replies")[0]  # 预期只有一个回复
    normalized = wrapper(reply)

    return normalized


## End of Pipeline
class Question(BaseModel):
    """
    表示单个推荐问题的Pydantic模型。

    属性:
        question: 推荐问题的文本
        category: 问题所属的类别
    """

    question: str
    category: str


class QuestionResult(BaseModel):
    """
    表示问题推荐完整结果的Pydantic模型。

    属性:
        questions: Question对象的列表
    """

    questions: list[Question]


# LLM的配置，确保它返回正确结构的JSON
QUESTION_RECOMMENDATION_MODEL_KWARGS = {
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "question_recommendation",
            "schema": QuestionResult.model_json_schema(),
        },
    }
}

# 指导LLM如何生成问题推荐的系统提示
system_prompt = """
You are an expert in data analysis and SQL query generation. Given a data model specification, optionally a user's question, and a list of categories, your task is to generate insightful, specific questions that can be answered using the provided data model. Each question should be accompanied by a brief explanation of its relevance or importance.

### JSON Output Structure

Output all questions in the following JSON format:

```json
{
    "questions": [
        {
            "question": "<generated question>",
            "category": "<category of the question>"
        },
        ...
    ]
}
```

### Guidelines for Generating Questions

1. **If Categories Are Provided:**

   - **Randomly select categories** from the list and ensure no single category dominates the output.
   - Ensure a balanced distribution of questions across all provided categories.
   - For each generated question, **randomize the category selection** to avoid a fixed order.

2. **Incorporate Diverse Analysis Techniques:**

   - Use a mix of the following analysis techniques for each category:
     - **Drill-down:** Delve into detailed levels of data.
     - **Roll-up:** Aggregate data to higher levels.
     - **Slice and Dice:** Analyze data from different perspectives.
     - **Trend Analysis:** Identify patterns or changes over time.
     - **Comparative Analysis:** Compare segments, groups, or time periods.

3. **If a User Question is Provided:**

   - Generate questions that are closely related to the user's previous question, ensuring that the new questions build upon or provide deeper insights into the original query.
   - Use **random category selection** to introduce diverse perspectives while maintaining a focus on the context of the previous question.
   - Apply the analysis techniques above to enhance the relevance and depth of the generated questions.

4. **If No User Question is Provided:**

   - Ensure questions cover different aspects of the data model.
   - Randomly distribute questions across all categories to ensure variety.

5. **General Guidelines for All Questions:**
   - Ensure questions can be answered using the data model.
   - Mix simple and complex questions.
   - Avoid open-ended questions – each should have a definite answer.
   - Incorporate time-based analysis where relevant.
   - Combine multiple analysis techniques when appropriate for deeper insights.

### Categories of Questions

1. **Descriptive Questions**
   Summarize historical data.

   - Example: _"What was the total sales volume for each product last quarter?"_

2. **Segmentation Questions**
   Identify meaningful data segments.

   - Example: _"Which customer segments contributed most to revenue growth?"_

3. **Comparative Questions**
   Compare data across segments or periods.

   - Example: _"How did Product A perform compared to Product B last year?"_

4. **Data Quality/Accuracy Questions**
   Assess data reliability and completeness.

   - Example: _"Are there inconsistencies in the sales records for Q1?"_

---

### Example JSON Output

```json
{
  "questions": [
    {
      "question": "What was the total revenue generated by each region in the last year?",
      "category": "Descriptive Questions"
    },
    {
      "question": "How do customer preferences differ between age groups?",
      "category": "Segmentation Questions"
    },
    {
      "question": "How does the conversion rate vary across different lead sources?",
      "category": "Comparative Questions"
    },
    {
      "question": "What percentage of contacts have incomplete or missing key properties (e.g., email, lifecycle stage, or deal association)",
      "category": "Data Quality/Accuracy Questions"
    }
  ]
}
```

---

### Additional Instructions for Randomization

- **Randomize Category Order:**
  Ensure that categories are selected in a random order for each question generation session.

- **Avoid Repetition:**
  Ensure the same category doesn't dominate the list by limiting the number of questions from any single category unless specified otherwise.

- **Diversity of Analysis:**
  Combine different analysis techniques (drill-down, roll-up, etc.) within the selected categories for richer insights.

- **Shuffle Categories:**
  If possible, shuffle the list of categories internally before generating questions to ensure varied selection.


"""

# 将填充动态值的用户提示模板
user_prompt_template = """
{% if models %}
Data Model Specification:
{{models}}
{% endif %}

{% if previous_questions %}
Previous Questions: {{previous_questions}}
{% endif %}

{% if categories %}
Categories: {{categories}}
{% endif %}

Current Date: {{current_date}}

Please generate {{max_questions}} insightful questions for each of the {{max_categories}} categories based on the provided data model. Both the questions and category names should be translated into {{language}}{% if user_question %} and be related to the user's question{% endif %}. The output format should maintain the structure but with localized text.
"""


class QuestionRecommendation(BasicPipeline):
    """
    基于数据模型和/或先前问题生成推荐问题的管道。

    该管道使用LLM生成有洞察力的、与上下文相关的问题，
    这些问题是用户可能想要询问有关其数据的。它可以生成纯粹基于
    数据模型结构的问题，或者可以基于先前的问题
    提供后续推荐。

    该管道由三个主要步骤组成:
    1. 使用数据模型和/或先前问题构建提示
    2. 使用LLM生成推荐
    3. 标准化和解析LLM的响应

    生成的问题被分类为不同类型（描述性、分段、
    比较、数据质量）并可以翻译成不同的语言。
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        **_,
    ):
        """
        初始化QuestionRecommendation管道。

        参数:
            llm_provider: 用于生成推荐的LLM提供者
            **_: 额外的关键字参数（被忽略）
        """
        # 设置管道所需的组件
        self._components = {
            "prompt_builder": PromptBuilder(template=user_prompt_template),
            "generator": llm_provider.get_generator(
                system_prompt=system_prompt,
                generation_kwargs=QUESTION_RECOMMENDATION_MODEL_KWARGS,
            ),
        }

        # 设置管道的最后一步
        self._final = "normalized"

        # 使用AsyncDriver初始化基础管道
        super().__init__(
            AsyncDriver({}, sys.modules[__name__], result_builder=base.DictResult())
        )

    @observe(name="Question Recommendation")
    async def run(
        self,
        mdl: dict,
        previous_questions: list[str] = [],
        categories: list[str] = [],
        language: str = "en",
        current_date: str = datetime.now().strftime("%Y-%m-%d %A %H:%M:%S"),
        max_questions: int = 5,
        max_categories: int = 3,
        **_,
    ) -> dict:
        """
        运行问题推荐管道。

        参数:
            mdl: 包含数据模型规范的字典
            previous_questions: 要在其基础上构建的先前问题列表
            categories: 要包含的问题类别列表
            language: 生成问题的目标语言（默认：英语）
            current_date: 上下文的当前日期字符串（默认：当前日期时间）
            max_questions: 每个类别要生成的最大问题数（默认：5）
            max_categories: 要包含的最大类别数（默认：3）
            **_: 额外的关键字参数（被忽略）

        返回:
            包含标准化问题推荐的字典
        """
        logger.info("Question Recommendation pipeline is running...")
        return await self._pipe.execute(
            [self._final],
            inputs={
                "mdl": mdl,
                "previous_questions": previous_questions,
                "categories": categories,
                "language": language,
                "current_date": current_date,
                "max_questions": max_questions,
                "max_categories": max_categories,
                **self._components,
            },
        )


if __name__ == "__main__":
    # 直接运行管道的代码，用于测试/开发目的
    from src.pipelines.common import dry_run_pipeline

    dry_run_pipeline(
        QuestionRecommendation,
        "question_recommendation",
        mdl={},
        previous_questions=[],
        categories=[],
        language="en",
        current_date=datetime.now().strftime("%Y-%m-%d %A %H:%M:%S"),
        max_questions=5,
        max_categories=3,
    )
