"""
平面图生成系统后端接口
基于FastAPI实现，提供配置保存、模型调用、任务管理等功能
"""

import os
import json
import uuid
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
USER_DATA_DIR = PROJECT_ROOT / "user_data"
OUTPUT_USER_DIR = PROJECT_ROOT / "output_user"
MODEL_DIR = PROJECT_ROOT / "model"
SCRIPT_PATH = PROJECT_ROOT / "script_user.sh"

# 确保目录存在
USER_DATA_DIR.mkdir(exist_ok=True)
OUTPUT_USER_DIR.mkdir(exist_ok=True)

# 任务状态管理
class TaskStatus:
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

# 任务信息存储
tasks: Dict[str, Dict] = {}
executor = ThreadPoolExecutor(max_workers=2)  # 限制并发任务数

# 数据模型
class TaskResponse(BaseModel):
    task_id: str
    status: str
    image_url: Optional[str] = None
    log_url: Optional[str] = None

class ApiResponse(BaseModel):
    code: int = Field(default=200, description="状态码：200成功，400客户端错误，500服务器错误")
    data: Optional[Dict] = Field(default=None, description="成功时返回数据，失败时为null")
    message: str = Field(default="success", description="提示信息")
    detail: str = Field(default="", description="详细错误信息（失败时）")

# 创建FastAPI应用
app = FastAPI(
    title="平面图生成系统API",
    description="房屋平面图生成系统的后端接口",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
app.mount("/api/user_data", StaticFiles(directory=str(USER_DATA_DIR)), name="user_data")
app.mount("/api/output", StaticFiles(directory=str(OUTPUT_USER_DIR)), name="output")

def create_api_response(code: int = 200, data: Optional[Dict] = None,
                        message: str = "success", detail: str = "") -> Dict:
    """创建标准API响应"""
    return {
        "code": code,
        "data": data,
        "message": message,
        "detail": detail
    }

def validate_config(config_json: Dict) -> bool:
    """验证配置数据的有效性"""
    try:
        # 检查必需字段
        required_fields = ["name", "room_type", "room_corner_nums", "room_area_rate"]
        for field in required_fields:
            if field not in config_json:
                return False

        # 检查数组长度一致性
        arrays = ["room_type", "room_corner_nums", "room_area_rate"]
        lengths = [len(config_json[field]) for field in arrays]
        if len(set(lengths)) != 1:
            return False

        # 检查面积占比总和是否为1（允许微小误差）
        # total_rate = sum(config_json["room_area_rate"])
        # if abs(total_rate - 1.0) > 0.001:
        #     return False

        # 检查拐点数量
        if any(corners < 4 for corners in config_json["room_corner_nums"]):
            return False

        # 检查前门数量
        front_door_count = config_json["room_type"].count(15)
        if front_door_count != 1:
            return False

        return True
    except Exception as e:
        logger.error(f"配置验证失败: {e}")
        return False

def execute_model_script(task_id: str, config_path: Path): # , boundary_image_path: Path
    """执行模型生成脚本"""
    task_output_dir = OUTPUT_USER_DIR / task_id
    task_output_dir.mkdir(exist_ok=True)

    log_file = task_output_dir / "log.txt"

    try:
        # 更新任务状态
        tasks[task_id]["status"] = TaskStatus.RUNNING
        tasks[task_id]["start_time"] = datetime.now()

        logger.info(f"任务 {task_id}: 开始执行模型脚本")

        # 构建命令
        cmd = [
            "bash", str(SCRIPT_PATH),
            "--demo_json", str(config_path),
            "--model_path", str(MODEL_DIR / "../ckpts/openai_2025_10_20_09_52_20_842787/model300000.pt"),
            "--output_path", str(task_output_dir),
            "--analog_bit", "False",
            "--dataset", "rplan",
            "--target_set", "8"
        ]

        logger.info(f"任务 {task_id}: 执行命令: {' '.join(cmd)}")

        # 执行命令，设置超时时间10分钟
        with open(log_file, "w") as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT
            )

            # 保存进程引用以便取消
            tasks[task_id]["process"] = process

            # 等待进程完成
            try:
                return_code = process.wait(timeout=600)  # 10分钟超时

                if return_code == 0:
                    # 检查输出文件是否存在
                    output_image = task_output_dir / "pred_b" / "0c_pred.png"
                    if output_image.exists():
                        tasks[task_id]["status"] = TaskStatus.SUCCESS
                        tasks[task_id]["image_url"] = f"/api/output/{task_id}/pred_b/0c_pred.png"
                        tasks[task_id]["log_url"] = f"/api/output/{task_id}/log.txt"
                        tasks[task_id]["end_time"] = datetime.now()
                        logger.info(f"任务 {task_id}: 执行成功")
                    else:
                        tasks[task_id]["status"] = TaskStatus.FAILED
                        tasks[task_id]["error"] = "模型执行成功但未生成输出图片"
                        logger.error(f"任务 {task_id}: 未找到输出图片")
                else:
                    tasks[task_id]["status"] = TaskStatus.FAILED
                    tasks[task_id]["error"] = f"脚本执行失败，返回码: {return_code}"
                    logger.error(f"任务 {task_id}: 脚本执行失败，返回码: {return_code}")

            except subprocess.TimeoutExpired:
                process.terminate()
                tasks[task_id]["status"] = TaskStatus.FAILED
                tasks[task_id]["error"] = "脚本执行超时（10分钟）"
                logger.error(f"任务 {task_id}: 脚本执行超时")

    except Exception as e:
        tasks[task_id]["status"] = TaskStatus.FAILED
        tasks[task_id]["error"] = str(e)
        logger.error(f"任务 {task_id}: 执行过程中发生异常: {e}", exc_info=True)

    finally:
        if "process" in tasks[task_id]:
            del tasks[task_id]["process"]

        if tasks[task_id]["status"] != TaskStatus.SUCCESS:
            tasks[task_id]["end_time"] = datetime.now()

@app.post("/api/generate-floor-plan", response_model=ApiResponse)
async def generate_floor_plan(
    background_tasks: BackgroundTasks,
    config: str = Form(...)
    # ,boundary_image: UploadFile = File(...)
):
    """
    生成平面图接口
    接收JSON配置和边界图，创建异步任务执行模型生成
    """
    try:
        # 解析配置JSON
        try:
            config_json = json.loads(config)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"JSON格式错误: {e}")

        # 验证配置
        if not validate_config(config_json):
            raise HTTPException(status_code=400, detail="配置数据验证失败")

        # 验证图片文件
        # if not boundary_image.content_type.startswith("image/"):
        #     raise HTTPException(status_code=400, detail="请上传图片文件")

        # 生成任务ID
        task_id = str(uuid.uuid4())

        # 保存配置文件
        config_path = USER_DATA_DIR / f"{task_id}.json"
        logger.info(f"正在保存json文件路径{config_path},内容：{config_json}")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_json, f, ensure_ascii=False, indent=2)

        # 保存边界图
        # boundary_image_path = USER_DATA_DIR / f"{task_id}.png"
        # with open(boundary_image_path, "wb") as f:
        #     content = await boundary_image.read()
        #     f.write(content)

        # 初始化任务信息
        tasks[task_id] = {
            "status": TaskStatus.PENDING,
            "config_path": str(config_path),
            # "boundary_image_path": str(boundary_image_path),
            "create_time": datetime.now(),
            "error": None,
            "image_url": None,
            "log_url": None
        }

        logger.info(f"任务 {task_id}: 创建成功，配置已保存")

        # 提交后台任务
        background_tasks.add_task(execute_model_script, task_id, config_path) #, boundary_image_path)

        return JSONResponse(
            status_code=200,
            content=create_api_response(
                data={"task_id": task_id},
                message="任务创建成功"
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")

@app.get("/api/task/{task_id}", response_model=ApiResponse)
async def get_task_status(task_id: str):
    """
    查询任务状态接口
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task_info = tasks[task_id]

    # 构建响应数据
    data = {
        "task_id": task_id,
        "status": task_info["status"]
    }

    if task_info["status"] == TaskStatus.SUCCESS:
        data["image_url"] = task_info.get("image_url")
        data["log_url"] = task_info.get("log_url")

    elif task_info["status"] == TaskStatus.FAILED:
        data["error"] = task_info.get("error", "未知错误")

    return JSONResponse(
        content=create_api_response(
            data=data,
            message="查询成功"
        )
    )

@app.post("/api/task/{task_id}/cancel", response_model=ApiResponse)
async def cancel_task(task_id: str):
    """
    取消任务接口
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task_info = tasks[task_id]

    if task_info["status"] not in [TaskStatus.PENDING, TaskStatus.RUNNING]:
        raise HTTPException(status_code=400, detail="任务无法取消（已完成或已失败）")

    # 如果任务正在运行，终止进程
    if task_info["status"] == TaskStatus.RUNNING and "process" in task_info:
        try:
            task_info["process"].terminate()
        except Exception as e:
            logger.error(f"终止进程失败: {e}")

    # 更新任务状态
    task_info["status"] = TaskStatus.CANCELLED
    task_info["end_time"] = datetime.now()

    logger.info(f"任务 {task_id}: 已取消")

    return JSONResponse(
        content=create_api_response(
            message="任务已取消"
        )
    )

@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    return JSONResponse(
        content=create_api_response(
            data={
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "project_root": str(PROJECT_ROOT),
                "user_data_dir": str(USER_DATA_DIR),
                "output_user_dir": str(OUTPUT_USER_DIR)
            }
        )
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content=create_api_response(
            code=exc.status_code,
            message="请求失败",
            detail=str(exc.detail)
        )
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=create_api_response(
            code=500,
            message="服务器内部错误",
            detail=str(exc)
        )
    )

if __name__ == "__main__":
    import uvicorn

    # 检查脚本文件是否存在
    if not SCRIPT_PATH.exists():
        logger.error(f"模型脚本不存在: {SCRIPT_PATH}")
        # 创建示例脚本
        SCRIPT_PATH.write_text("""#!/bin/bash
echo "模型脚本占位符"
echo "实际脚本路径: $PROJECT_ROOT/script_user.sh"
echo "请根据实际模型调用方式修改此脚本"
""")
        SCRIPT_PATH.chmod(0o755)
        logger.info(f"已创建示例脚本: {SCRIPT_PATH}")

    # 检查模型文件是否存在
    model_path = MODEL_DIR / "../ckpts/openai_2025_10_20_09_52_20_842787/model300000.pt"
    if not model_path.exists():
        logger.warning(f"模型文件不存在: {model_path}")
        logger.warning("请确保模型文件已正确放置")

    logger.info("启动平面图生成系统后端服务")
    logger.info(f"项目根目录: {PROJECT_ROOT}")
    logger.info(f"用户数据目录: {USER_DATA_DIR}")
    logger.info(f"输出目录: {OUTPUT_USER_DIR}")
    logger.info(f"API文档: http://192.168.200.151:8000/docs")

    uvicorn.run(
        "main:app",
        host="192.168.200.151",
        port=8000,
        reload=False,
        log_level="info"
    )