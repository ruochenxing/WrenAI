import asyncio
import logging
from typing import Dict, Literal, Optional

from cachetools import TTLCache
from langfuse.decorators import observe
from pydantic import AliasChoices, BaseModel, Field

from src.core.pipeline import BasicPipeline
from src.utils import trace_metadata

logger = logging.getLogger("wren-ai-service")

# 这个文件的主要功能是：
# 语义准备服务：
#   - 接收数据模型定义(MDL)并进行预处理
#   - 并行执行多个语义分析管道
#   - 为后续的语义描述和查询做准备
# 处理流程：
#   - 数据库模式分析(db_schema)
#   - 历史问题分析(historical_question)
#   - 表描述生成(table_description)
#   - SQL示例对生成(sql_pairs)
# 状态管理：
#   - 跟踪处理状态（indexing/finished/failed）
#   - 提供状态查询接口
#   - 错误处理和日志记录
# 资源管理：
#   - 支持资源清理（删除语义文档）
#   - 使用TTL缓存优化性能
#   - 项目级别的资源隔离


# POST /v1/semantics-preparations
class SemanticsPreparationRequest(BaseModel):
    """语义准备请求模型

    用于接收初始化语义准备的请求参数
    """

    mdl: str  # 数据模型定义(MDL)字符串
    # 不建议使用id作为字段名，但由于API规范中使用了，所以需要支持，未来会移除
    mdl_hash: str = Field(
        validation_alias=AliasChoices("mdl_hash", "id")
    )  # MDL的哈希值，用于唯一标识
    project_id: Optional[str] = None  # 项目ID，用于资源隔离


class SemanticsPreparationResponse(BaseModel):
    """语义准备响应模型

    用于返回语义准备请求的处理结果
    """

    # 不建议使用id作为字段名，但由于API规范中使用了，所以需要支持，未来会移除
    mdl_hash: str = Field(serialization_alias="id")  # MDL的哈希值，用于后续状态查询


# GET /v1/semantics-preparations/{mdl_hash}/status
class SemanticsPreparationStatusRequest(BaseModel):
    """语义准备状态查询请求模型

    用于查询特定MDL的语义准备状态
    """

    # 不建议使用id作为字段名，但由于API规范中使用了，所以需要支持，未来会移除
    mdl_hash: str = Field(
        validation_alias=AliasChoices("mdl_hash", "id")
    )  # MDL的哈希值


class SemanticsPreparationStatusResponse(BaseModel):
    """语义准备状态响应模型

    用于返回语义准备的当前状态
    """

    class SemanticsPreparationError(BaseModel):
        """错误信息模型"""

        code: Literal["OTHERS"]  # 错误代码
        message: str  # 错误详细信息

    status: Literal["indexing", "finished", "failed"]  # 处理状态：索引中、已完成、失败
    error: Optional[SemanticsPreparationError] = None  # 错误信息，仅在失败时存在


class SemanticsPreparationService:
    """语义准备服务类

    负责协调和执行数据模型的语义准备工作，包括并行处理多个语义分析管道
    """

    def __init__(
        self,
        pipelines: Dict[str, BasicPipeline],  # 处理管道组件字典
        maxsize: int = 1_000_000,  # 缓存最大条目数
        ttl: int = 120,  # 缓存条目过期时间(秒)
    ):
        """初始化语义准备服务

        Args:
            pipelines: 包含各种处理管道的字典
            maxsize: 缓存的最大容量
            ttl: 缓存条目的过期时间
        """
        self._pipelines = pipelines
        self._prepare_semantics_statuses: Dict[
            str, SemanticsPreparationStatusResponse
        ] = TTLCache(maxsize=maxsize, ttl=ttl)

    @observe(name="Prepare Semantics")
    @trace_metadata
    async def prepare_semantics(
        self,
        prepare_semantics_request: SemanticsPreparationRequest,
        **kwargs,
    ):
        """准备语义数据

        执行多个并行管道来处理MDL数据，包括：
        - 数据库模式分析
        - 历史问题分析
        - 表描述生成
        - SQL示例对生成

        Args:
            prepare_semantics_request: 包含MDL等信息的请求对象
        """
        results = {
            "metadata": {
                "error_type": "",
                "error_message": "",
            },
        }

        try:
            logger.info(f"MDL: {prepare_semantics_request.mdl}")

            input = {
                "mdl_str": prepare_semantics_request.mdl,
                "project_id": prepare_semantics_request.project_id,
            }

            # 并行执行多个处理管道
            tasks = [
                self._pipelines[name].run(**input)
                for name in [
                    "db_schema",  # 数据库模式分析
                    "historical_question",  # 历史问题分析
                    "table_description",  # 表描述生成
                    "sql_pairs",  # SQL示例对生成
                ]
            ]

            await asyncio.gather(*tasks)

            # 更新处理状态为完成
            self._prepare_semantics_statuses[
                prepare_semantics_request.mdl_hash
            ] = SemanticsPreparationStatusResponse(
                status="finished",
            )
        except Exception as e:
            logger.exception(f"Failed to prepare semantics: {e}")

            # 更新处理状态为失败
            self._prepare_semantics_statuses[
                prepare_semantics_request.mdl_hash
            ] = SemanticsPreparationStatusResponse(
                status="failed",
                error=SemanticsPreparationStatusResponse.SemanticsPreparationError(
                    code="OTHERS",
                    message=f"Failed to prepare semantics: {e}",
                ),
            )

            results["metadata"]["error_type"] = "INDEXING_FAILED"
            results["metadata"]["error_message"] = str(e)

        return results

    def get_prepare_semantics_status(
        self, prepare_semantics_status_request: SemanticsPreparationStatusRequest
    ) -> SemanticsPreparationStatusResponse:
        """获取语义准备的状态

        Args:
            prepare_semantics_status_request: 状态查询请求

        Returns:
            当前的处理状态，如果找不到对应的请求则返回失败状态
        """
        if (
            result := self._prepare_semantics_statuses.get(
                prepare_semantics_status_request.mdl_hash
            )
        ) is None:
            logger.exception(
                f"id is not found for SemanticsPreparation: {prepare_semantics_status_request.mdl_hash}"
            )
            return SemanticsPreparationStatusResponse(
                status="failed",
                error=SemanticsPreparationStatusResponse.SemanticsPreparationError(
                    code="OTHERS",
                    message="{prepare_semantics_status_request.id} is not found",
                ),
            )

        return result

    @observe(name="Delete Semantics Documents")
    @trace_metadata
    async def delete_semantics(self, project_id: str):
        """删除项目相关的语义文档

        清理指定项目的所有语义分析结果

        Args:
            project_id: 要清理的项目ID
        """
        logger.info(f"Project ID: {project_id}, Deleting semantics documents...")

        # 并行清理多个管道的数据
        tasks = [
            self._pipelines[name].clean(project_id=project_id)
            for name in ["db_schema", "historical_question", "table_description"]
        ] + [
            self._pipelines["sql_pairs"].clean(
                sql_pairs=[],
                project_id=project_id,
                delete_all=True,
            )
        ]

        await asyncio.gather(*tasks)
