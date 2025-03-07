from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, RedirectResponse

from src.config import settings
from src.globals import (
    create_service_container,
    create_service_metadata,
)
from src.providers import generate_components
from src.utils import (
    setup_custom_logger,
)
from src.web.v1 import routers

# 设置自定义日志记录器
setup_custom_logger(
    "wren-ai-service", level_str=settings.logging_level, is_dev=settings.development
)

# 这个文件是 WrenAI 服务的主入口文件，主要功能包括：
# 服务配置：
# 设置日志记录
# 配置 FastAPI 应用
# 设置 CORS 策略
# 生命周期管理：
# 启动时初始化组件和服务
# 关闭时清理资源
# 路由管理：
# API 路由注册
# 开发路由（仅在开发模式）
# 错误处理：
# 全局异常处理
# 请求验证错误处理
# 基础端点：
# 根路由重定向到文档
# 健康检查端点
# 服务器配置：
# 使用 uvicorn 作为 ASGI 服务器
# 配置热重载和文件监视
# 设置工作进程和HTTP实现
# 这个文件是整个服务的核心启动点，负责将所有组件组织在一起并启动服务。


# FastAPI的生命周期管理
# https://fastapi.tiangolo.com/advanced/events/#lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动事件
    # 生成管道组件
    print("settings", settings)
    pipe_components = generate_components(settings.components)
    # 创建服务容器和元数据
    app.state.service_container = create_service_container(pipe_components, settings)
    app.state.service_metadata = create_service_metadata(pipe_components)
    # 初始化langfuse（用于监控和跟踪）
    # init_langfuse(settings)

    yield

    # 关闭事件
    # 刷新langfuse上下文
    # langfuse_context.flush()


# 创建FastAPI应用实例
app = FastAPI(
    title="wren-ai-service API Docs",  # API文档标题
    lifespan=lifespan,  # 设置生命周期管理器
    redoc_url=None,  # 禁用redoc文档
    default_response_class=ORJSONResponse,  # 设置默认响应类为ORJSON
)

# 添加CORS中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,  # 允许携带凭证
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

# 注册API路由
app.include_router(routers.router, prefix="/v1", tags=["v1"])

# TODO: 已废弃，仅用于locust负载测试，将来会移除
if settings.development:
    from src.web import development

    app.include_router(development.router, prefix="/dev", tags=["dev"])


# 全局异常处理器
@app.exception_handler(Exception)
async def exception_handler(_, exc: Exception):
    # 处理所有未捕获的异常，返回500错误
    return ORJSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


# 请求验证异常处理器
@app.exception_handler(RequestValidationError)
async def request_exception_handler(_, exc: Exception):
    # 处理请求验证错误，返回400错误
    return ORJSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


# 根路由处理器
@app.get("/")
def root():
    # 重定向到API文档页面
    return RedirectResponse(url="/docs")


# 健康检查端点
@app.get("/health")
def health():
    # 返回服务状态
    return {"status": "ok"}


# 主程序入口
if __name__ == "__main__":
    # 启动uvicorn服务器
    uvicorn.run(
        "src.__main__:app",
        host=settings.host,  # 主机地址
        port=settings.port,  # 端口号
        reload=settings.development,  # 开发模式下启用热重载
        reload_includes=[],  # 监视这些文件变化
        reload_excludes=[
            "demo/*.py",
            "tests/**/*.py",
            "eval/**/*.py",
        ],  # 排除这些文件的监视
        workers=1,  # 工作进程数
        loop="none",  # 不指定事件循环
        http="httptools",  # 使用httptools作为HTTP协议实现
    )
