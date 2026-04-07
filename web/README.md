# 平面图生成系统

基于深度学习的房屋平面图生成系统，支持用户自定义户型配置并生成平面图。

## 系统架构

- **前端**: Vue3 + Element Plus + Vite
- **后端**: FastAPI + Python 3.9+
- **模型**: HouseDiffusion 平面图生成模型

## 功能特性

1. **可视化配置**: 通过Web界面配置房间属性、连接关系
2. **实时预览**: JSON配置和拓扑图实时预览
3. **异步生成**: 后台执行模型生成，支持任务状态查询
4. **文件管理**: 自动保存配置和生成结果
5. **响应式设计**: 适配PC端主流分辨率

## 快速开始

### 环境要求

- Python 3.9+
- Node.js 16+ (仅前端开发需要)
- Bash shell (模型脚本执行)

### 后端部署

1. **安装依赖**:
```bash
cd web
pip install -r requirements.txt
```

2. **配置模型路径**:
确保模型文件位于正确位置:
```
C2Plan/ckpts/openai_2025_10_20_09_52_20_842787/model300000.pt
```

3. **启动后端服务**:
```bash
cd web
python main.py
```
或使用uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

服务启动后访问:
- API文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/api/health

### 前端使用

前端为纯静态页面，无需构建，直接使用浏览器打开:

1. **直接打开**:
双击 `web/index.html` 在浏览器中打开

2. **或使用HTTP服务器**:
```bash
cd web
python -m http.server 8080
```
然后访问: http://localhost:8080

## 目录结构

```
web/
├── index.html          # 前端主页面
├── style.css           # 样式文件
├── app.js             # 前端逻辑
├── main.py            # 后端FastAPI应用
├── requirements.txt    # Python依赖
└── README.md          # 说明文档

C2Plan/
├── script_user.sh     # 模型调用脚本
├── user_data/         # 用户配置和边界图
├── output_user/       # 模型输出结果
└── model/            # 模型文件
```

## API接口

### 1. 生成平面图
- **URL**: `POST /api/generate-floor-plan`
- **Content-Type**: `multipart/form-data`
- **参数**:
  - `config`: JSON格式的户型配置
  - `boundary_image`: 边界图图片文件
- **响应**: 返回任务ID

### 2. 查询任务状态
- **URL**: `GET /api/task/{task_id}`
- **响应**: 返回任务状态和结果信息

### 3. 取消任务
- **URL**: `POST /api/task/{task_id}/cancel`
- **响应**: 取消指定任务

### 4. 静态文件访问
- 用户配置: `/api/user_data/{filename}`
- 输出结果: `/api/output/{task_id}/{filename}`

## 配置数据格式

```json
{
    "name": "户型名称",
    "room_type": [1, 2, 3, 15, 17, ...],
    "room_corner_nums": [4, 4, 4, 4, 4, ...],
    "room_area_rate": [0.3, 0.2, 0.1, 0.02, 0.02, ...],
    "room_connections": [[0, 3], [1, 3], [2, 4], ...]
}
```

**房间类型映射**:
- 1: 客厅, 2: 厨房, 3: 卧室, 4: 卫生间, 5: 阳台
- 6: 通道, 7: 餐厅, 8: 书房, 10: 储藏室
- 15: 前门, 17: 内门

## 校验规则

1. 面积占比总和必须等于1
2. 所有房间拐点数量≥4
3. 必须有且仅有1个前门（类型15）
4. 内门（类型17）随连接关系自动生成
5. 数组长度必须一致

## 模型脚本配置

系统使用 `script_user.sh` 调用模型，默认配置如下:

```bash
python user_data/run_demo.py \
    --demo_json model/demo.json \
    --model_path ckpts/openai_2025_10_20_09_52_20_842787/model300000.pt \
    --output_path ./output_user \
    --analog_bit False \
    --dataset rplan \
    --target_set 8
```

**注意**: 请根据实际模型位置修改脚本中的路径。

## 常见问题

### 1. 模型文件不存在
**错误**: `模型文件不存在: .../model300000.pt`
**解决**: 确保模型文件已正确放置到指定位置

### 2. 脚本执行权限不足
**错误**: `Permission denied`
**解决**: 给脚本添加执行权限
```bash
chmod +x C2Plan/script_user.sh
```

### 3. 端口被占用
**错误**: `Address already in use`
**解决**: 修改端口号
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### 4. CORS错误
**错误**: 前端无法访问后端API
**解决**: 确保后端CORS配置正确，或使用同源部署

### 5. 文件保存失败
**错误**: `Permission denied` 或 `No such file or directory`
**解决**: 确保 `user_data` 和 `output_user` 目录有写入权限

## 开发说明

### 前端开发
如需修改前端代码:
1. 编辑 `index.html`, `style.css`, `app.js`
2. 刷新浏览器查看效果

### 后端开发
1. 修改 `main.py` 中的业务逻辑
2. 重启服务使更改生效
3. 查看日志输出调试信息

### 模型集成
如需修改模型调用方式:
1. 编辑 `C2Plan/script_user.sh`
2. 确保脚本接受正确的参数
3. 输出结果保存到指定目录

## 性能优化建议

1. **并发控制**: 默认限制2个并发任务，可根据服务器性能调整
2. **超时设置**: 模型执行超时时间为10分钟
3. **日志轮转**: 建议配置日志轮转避免日志文件过大
4. **资源清理**: 定期清理旧的输出文件

## 安全注意事项

1. **生产环境**:
   - 限制CORS来源
   - 添加身份验证
   - 配置HTTPS
   - 设置文件上传大小限制

2. **文件安全**:
   - 验证上传文件类型
   - 限制文件大小
   - 防止路径遍历攻击

## 许可证

本项目仅供学习和研究使用。

## 技术支持

如有问题，请检查:
1. 控制台错误信息
2. 后端服务日志
3. 模型执行日志 (`output_user/{task_id}/log.txt`)