const { createApp, ref, computed, watch, onMounted } = Vue;
const { ElMessage, ElMessageBox } = ElementPlus;
   
createApp({
    setup() {
        // 房间类型选项
        const roomTypeOptions = [
            { value: 1, label: '1 - 客厅' },
            { value: 2, label: '2 - 厨房' },
            { value: 3, label: '3 - 卧室' },
            { value: 4, label: '4 - 卫生间' },
            { value: 5, label: '5 - 阳台' },
            { value: 6, label: '6 - 通道' },
            { value: 7, label: '7 - 餐厅' },
            { value: 8, label: '8 - 书房' },
            { value: 10, label: '10 - 储藏室' },
            { value: 15, label: '15 - 前门' },
            { value: 17, label: '17 - 内门' }
        ];

        // 响应式数据
        const rooms = ref([
            { type: 1, corners: 4, areaRate: 0.28 },  // 客厅
            { type: 2, corners: 4, areaRate: 0.1 },   // 厨房
            { type: 3, corners: 4, areaRate: 0.2 },   // 卧室
            { type: 15, corners: 4, areaRate: 0.02 }  // 前门（必须）
        ]);

        const connections = ref([]);
        const connectionRoomA = ref(null);
        const connectionRoomB = ref(null);
        const boundaryImage = ref(null);
        const boundaryImagePreview = ref(null);
        const boundaryFileName = ref('');
        const dialogVisible = ref(false);
        const dialogTitle = ref('');
        const dialogMessage = ref('');
        const dialogCallback = ref(null);
        const generationStatus = ref('idle'); // idle, generating, success, failed
        const generationProgress = ref(0);
        const generationMessage = ref('');
        const generatedImageUrl = ref('');
        const errorMessage = ref('');
        const taskId = ref('');
        const checkInterval = ref(null);
        const isGenerating = ref(false);

        // 拓扑图相关
        const topologyWidth = ref(400);
        const topologyHeight = ref(300);
        const tooltip = ref({
            visible: false,
            x: 0,
            y: 0,
            content: ''
        });

        // 计算属性
        const hasFrontDoor = computed(() => {
            return rooms.value.some(room => room.type === 15);
        });

        const nonDoorRooms = computed(() => {
            return rooms.value
                .map((room, index) => ({ ...room, index }))
                .filter(room => room.type !== 15 && room.type !== 17);
        });

        const canCreateConnection = computed(() => {
            return connectionRoomA.value !== null &&
                   connectionRoomB.value !== null &&
                   connectionRoomA.value !== connectionRoomB.value;
        });

        const validationErrors = ref([]);

        const isValid = computed(() => {
            validationErrors.value = [];

            // 检查面积占比总和是否为1
            const totalAreaRate = rooms.value.reduce((sum, room) => sum + room.areaRate, 0);
            const areaValid = Math.abs(totalAreaRate - 1) < 0.001;
            if (!areaValid) {
                validationErrors.value.push(`面积占比总和应为1，当前为${totalAreaRate.toFixed(3)}`);
            }

            // 检查拐点数量
            const cornersValid = rooms.value.every(room => room.corners >= 4);
            if (!cornersValid) {
                validationErrors.value.push('所有房间拐点数量必须≥4');
            }

            // 检查前门数量
            const frontDoorCount = rooms.value.filter(room => room.type === 15).length;
            const frontDoorValid = frontDoorCount === 1;
            if (!frontDoorValid) {
                if (frontDoorCount === 0) {
                    validationErrors.value.push('必须设置1个前门');
                } else {
                    validationErrors.value.push(`只能有1个前门，当前有${frontDoorCount}个`);
                }
            }

            // 检查边界图
            const boundaryValid = boundaryImage.value !== null;
            if (!boundaryValid) {
                validationErrors.value.push('请上传边界图');
            }

            // 临时允许没有边界图进行测试
            const testMode = true; // 设置为false恢复严格校验
            if (testMode) {
                return areaValid && cornersValid && frontDoorValid;
            }

            return areaValid && cornersValid && frontDoorValid && boundaryValid;
        });

        const generatedJson = computed(() => {
            // 生成房间类型数组
            const roomType = rooms.value.map(room => room.type);
            const roomCornerNums = rooms.value.map(room => room.corners);
            const roomAreaRate = rooms.value.map(room => room.areaRate);

            // 生成连接关系
            const roomConnections = connections.value.map(conn => [
                [conn.roomA, conn.doorIndex],
                [conn.roomB, conn.doorIndex]
            ]).flat();

            return {
                name: boundaryFileName.value.replace(/\.[^/.]+$/, '') || 'untitled',
                room_type: roomType,
                room_corner_nums: roomCornerNums,
                room_area_rate: roomAreaRate,
                room_connections: roomConnections
            };
        });

        const formattedJson = computed(() => {
            return formatJsonWithHighlight(generatedJson.value);
        });

        // JSON编辑相关
        const jsonText = ref('');
        const jsonError = ref('');
        const jsonEditTimeout = ref(null);

        // 监听generatedJson变化，更新jsonText
        watch(generatedJson, (newJson) => {
            if (jsonEditTimeout.value) {
                clearTimeout(jsonEditTimeout.value);
            }

            // 延迟更新，避免与用户输入冲突
            jsonEditTimeout.value = setTimeout(() => {
                try {
                    const currentText = jsonText.value.trim();
                    if (currentText) {
                        // 尝试解析当前文本，如果与newJson不同才更新
                        const currentJson = JSON.parse(currentText);
                        if (JSON.stringify(currentJson) !== JSON.stringify(newJson)) {
                            jsonText.value = JSON.stringify(newJson, null, 2);
                            jsonError.value = '';
                        }
                    } else {
                        jsonText.value = JSON.stringify(newJson, null, 2);
                        jsonError.value = '';
                    }
                } catch (e) {
                    // 如果当前文本不是有效JSON，直接更新
                    jsonText.value = JSON.stringify(newJson, null, 2);
                    jsonError.value = '';
                }
            }, 100);
        }, { deep: true, immediate: true });

        const onJsonInput = () => {
            if (jsonEditTimeout.value) {
                clearTimeout(jsonEditTimeout.value);
            }

            jsonEditTimeout.value = setTimeout(() => {
                validateJson();
            }, 500);
        };

        const validateJson = () => {
            try {
                if (!jsonText.value.trim()) {
                    jsonError.value = '';
                    return false;
                }

                const parsed = JSON.parse(jsonText.value);

                // 基本结构验证
                const requiredFields = ['name', 'room_type', 'room_corner_nums', 'room_area_rate', 'room_connections'];
                for (const field of requiredFields) {
                    if (!(field in parsed)) {
                        jsonError.value = `缺少必需字段: ${field}`;
                        return false;
                    }
                }

                // 数组长度验证
                const arrays = ['room_type', 'room_corner_nums', 'room_area_rate'];
                const lengths = arrays.map(field => parsed[field].length);
                if (new Set(lengths).size !== 1) {
                    jsonError.value = '数组长度不一致';
                    return false;
                }

                jsonError.value = '';
                return true;
            } catch (error) {
                jsonError.value = `JSON格式错误: ${error.message}`;
                return false;
            }
        };

        const formatJson = () => {
            try {
                if (!jsonText.value.trim()) {
                    jsonText.value = JSON.stringify(generatedJson.value, null, 2);
                    return;
                }

                const parsed = JSON.parse(jsonText.value);
                jsonText.value = JSON.stringify(parsed, null, 2);
                jsonError.value = '';
                ElMessage.success('JSON已格式化');
            } catch (error) {
                jsonError.value = `格式化失败: ${error.message}`;
                ElMessage.error('JSON格式化失败');
            }
        };

        const updateFromJson = () => {
            if (!validateJson()) {
                ElMessage.error('JSON格式错误，请先修正');
                return;
            }

            try {
                const parsed = JSON.parse(jsonText.value);

                // 验证数据
                if (!validateJsonData(parsed)) {
                    ElMessage.error('JSON数据验证失败');
                    return;
                }

                // 更新房间数据
                const roomCount = parsed.room_type.length;
                const newRooms = [];

                for (let i = 0; i < roomCount; i++) {
                    newRooms.push({
                        type: parsed.room_type[i],
                        corners: parsed.room_corner_nums[i],
                        areaRate: parsed.room_area_rate[i]
                    });
                }

                // 更新连接关系
                const newConnections = [];
                const doorIndices = new Set();

                // 解析连接关系
                for (let i = 0; i < parsed.room_connections.length; i += 2) {
                    const connA = parsed.room_connections[i];
                    const connB = parsed.room_connections[i + 1];

                    if (connA && connB) {
                        const roomA = connA[0];
                        const doorIndex = connA[1];
                        const roomB = connB[0];

                        // 验证门索引
                        if (doorIndex >= roomCount || parsed.room_type[doorIndex] !== 17) {
                            ElMessage.error(`连接关系错误: 索引${doorIndex}不是内门`);
                            return;
                        }

                        newConnections.push({
                            roomA: roomA,
                            roomB: roomB,
                            doorIndex: doorIndex
                        });
                        doorIndices.add(doorIndex);
                    }
                }

                // 更新边界图文件名
                const newBoundaryFileName = parsed.name + '.png';

                // 应用更新
                rooms.value = newRooms;
                connections.value = newConnections;

                // 更新边界图文件名（如果已上传的图片名称匹配）
                if (boundaryFileName.value && boundaryFileName.value.replace(/\.[^/.]+$/, '') !== parsed.name) {
                    ElMessage.info(`JSON中的名称"${parsed.name}"与当前边界图名称不匹配`);
                }

                jsonError.value = '';
                ElMessage.success('配置已从JSON更新');

                // 滚动到顶部
                window.scrollTo({ top: 0, behavior: 'smooth' });

            } catch (error) {
                jsonError.value = `更新失败: ${error.message}`;
                ElMessage.error('更新配置失败');
            }
        };

        const validateJsonData = (json) => {
            // 检查面积占比总和
            const totalAreaRate = json.room_area_rate.reduce((sum, rate) => sum + rate, 0);
            if (Math.abs(totalAreaRate - 1) > 0.001) {
                jsonError.value = `面积占比总和应为1，当前为${totalAreaRate.toFixed(3)}`;
                return false;
            }

            // 检查拐点数量
            if (json.room_corner_nums.some(corners => corners < 4)) {
                jsonError.value = '所有房间拐点数量必须≥4';
                return false;
            }

            // 检查前门数量
            const frontDoorCount = json.room_type.filter(type => type === 15).length;
            if (frontDoorCount !== 1) {
                jsonError.value = `必须有且仅有1个前门，当前有${frontDoorCount}个`;
                return false;
            }

            return true;
        };

        const topologyNodes = computed(() => {
            const nodes = [];
            const nonDoor = nonDoorRooms.value;
            const centerX = topologyWidth.value / 2;
            const centerY = topologyHeight.value / 2;

            if (nonDoor.length === 0) return nodes;

            // 按房间索引排序，确保位置稳定
            const sortedNonDoor = [...nonDoor].sort((a, b) => a.index - b.index);

            // 找到最大和最小的面积占比，用于归一化
            const areaRates = sortedNonDoor.map(room => room.areaRate);
            const maxAreaRate = Math.max(...areaRates);
            const minAreaRate = Math.min(...areaRates);

            // 基础半径和缩放因子
            const baseRadius = 15; // 最小半径
            const scaleFactor = 25; // 缩放因子，控制最大半径

            // 计算最大可能半径，用于自适应布局
            const maxRoomRadius = baseRadius + scaleFactor;

            // 计算布局半径：根据容器大小和房间数量自适应
            // 确保所有房间（包括半径）都在画布内
            const containerPadding = 20; // 容器内边距
            const availableWidth = topologyWidth.value - 2 * containerPadding - 2 * maxRoomRadius;
            const availableHeight = topologyHeight.value - 2 * containerPadding - 2 * maxRoomRadius;
            const layoutRadius = Math.min(availableWidth, availableHeight) * 0.4;

            // 计算房间节点位置（圆形布局）
            sortedNonDoor.forEach((room, arrayIndex) => {
                // 计算角度
                const angle = (arrayIndex / Math.max(sortedNonDoor.length, 1)) * 2 * Math.PI;
                const x = centerX + Math.cos(angle) * layoutRadius;
                const y = centerY + Math.sin(angle) * layoutRadius;

                // 根据面积占比计算半径
                // 如果所有房间面积相同，使用基础半径
                let radius;
                if (maxAreaRate === minAreaRate) {
                    radius = baseRadius + scaleFactor / 2;
                } else {
                    // 归一化面积占比到 [0, 1] 范围
                    const normalizedArea = (room.areaRate - minAreaRate) / (maxAreaRate - minAreaRate);
                    // 计算半径：基础半径 + 缩放因子 * 归一化面积
                    radius = baseRadius + scaleFactor * normalizedArea;
                }

                // 确保房间在画布内
                const adjustedX = Math.max(radius + containerPadding, Math.min(topologyWidth.value - radius - containerPadding, x));
                const adjustedY = Math.max(radius + containerPadding, Math.min(topologyHeight.value - radius - containerPadding, y));

                nodes.push({
                    id: `room-${room.index}`,
                    type: 'room',
                    index: room.index,
                    roomType: room.type,
                    areaRate: room.areaRate,
                    x: adjustedX,
                    y: adjustedY,
                    radius: radius,
                    color: getRoomColor(room.type)
                });
            });

            return nodes;
        });

        const topologyDoors = computed(() => {
            const doors = [];
            const doorRooms = rooms.value
                .map((room, index) => ({ ...room, index }))
                .filter(room => room.type === 17);

            // 计算门节点位置（在连接的两个房间之间的连线上，考虑房间半径）
            connections.value.forEach((conn, index) => {
                const roomA = topologyNodes.value.find(n => n.index === conn.roomA);
                const roomB = topologyNodes.value.find(n => n.index === conn.roomB);

                if (roomA && roomB) {
                    // 计算两个房间中心点的向量
                    const dx = roomB.x - roomA.x;
                    const dy = roomB.y - roomA.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);

                    if (distance > 0) {
                        // 计算单位向量
                        const unitX = dx / distance;
                        const unitY = dy / distance;

                        // 计算门的位置：在两个房间之间的连线上
                        // 考虑房间半径，让门位于两个房间边缘之间
                        const totalRadius = roomA.radius + roomB.radius;
                        const overlap = Math.max(0, totalRadius - distance);

                        // 如果房间有重叠，调整位置
                        const adjustedDistance = distance + overlap;

                        // 门距离房间A的比例（考虑房间半径）
                        const ratioFromA = roomA.radius / totalRadius;

                        // 计算门的位置
                        const x = roomA.x + unitX * (ratioFromA * adjustedDistance);
                        const y = roomA.y + unitY * (ratioFromA * adjustedDistance);

                        // 门的大小根据连接的房间大小调整（取平均值）
                        const avgRadius = (roomA.radius + roomB.radius) / 2;
                        const doorRadius = Math.max(8, Math.min(15, avgRadius * 0.35));

                        // 确保门在画布内
                        const containerPadding = 10;
                        const adjustedDoorX = Math.max(doorRadius + containerPadding, Math.min(topologyWidth.value - doorRadius - containerPadding, x));
                        const adjustedDoorY = Math.max(doorRadius + containerPadding, Math.min(topologyHeight.value - doorRadius - containerPadding, y));

                        doors.push({
                            id: `door-${conn.doorIndex}`,
                            type: 'door',
                            index: conn.doorIndex,
                            x: adjustedDoorX,
                            y: adjustedDoorY,
                            radius: doorRadius,
                            color: '#D3A2C7'  // 内门颜色
                        });
                    }
                }
            });

            return doors;
        });

        const topologyLinks = computed(() => {
            const links = [];

            connections.value.forEach(conn => {
                const roomA = topologyNodes.value.find(n => n.index === conn.roomA);
                const roomB = topologyNodes.value.find(n => n.index === conn.roomB);
                const door = topologyDoors.value.find(d => d.index === conn.doorIndex);

                if (roomA && door) {
                    links.push({
                        id: `link-${conn.roomA}-${conn.doorIndex}`,
                        source: roomA,
                        target: door,
                        color: '#D3A2C7'  // 连接线颜色与内门一致
                    });
                }

                if (roomB && door) {
                    links.push({
                        id: `link-${conn.roomB}-${conn.doorIndex}`,
                        source: roomB,
                        target: door,
                        color: '#D3A2C7'  // 连接线颜色与内门一致
                    });
                }
            });

            return links;
        });

        // 方法
        const getRoomColor = (type) => {
            const ID_COLOR = {
                1: '#EE4D4D',   // 客厅
                2: '#C67C7B',   // 厨房
                3: '#FFD274',   // 卧室
                4: '#BEBEBE',   // 卫生间
                5: '#BFE3E8',   // 阳台
                6: '#7BA779',   // 通道
                7: '#E87A90',   // 餐厅
                8: '#FF8C69',   // 书房
                10: '#1F849B',  // 储藏室
                15: '#727171',  // 前门
                17: '#D3A2C7'   // 内门
            };
            return ID_COLOR[type] || '#999999';
        };

        const addRoom = () => {
            rooms.value.push({ type: 1, corners: 4, areaRate: 0.1 });
        };

        const removeRoom = (index) => {
            if (index < rooms.value.length - 1) {
                // 检查是否删除的是门
                const room = rooms.value[index];
                if (room.type === 15 || room.type === 17) {
                    showDialog('提示', '不能删除门房间，请先删除对应的连接关系', 'warning');
                    return;
                }

                // 更新连接关系中的房间索引
                connections.value = connections.value.filter(conn =>
                    conn.roomA !== index && conn.roomB !== index
                ).map(conn => ({
                    ...conn,
                    roomA: conn.roomA > index ? conn.roomA - 1 : conn.roomA,
                    roomB: conn.roomB > index ? conn.roomB - 1 : conn.roomB
                }));

                rooms.value.splice(index, 1);
            }
        };

        const setAsFrontDoor = (index) => {
            if (!hasFrontDoor.value) {
                rooms.value[index].type = 15;
                rooms.value[index].corners = 4;
                rooms.value[index].areaRate = 0.02;
                ElMessage.success('已设置为前门');
            } else {
                ElMessage.warning('只能设置一个前门');
            }
        };

        const calculateProportions = () => {
            const nonDoorRoomsCount = rooms.value.filter(room =>
                room.type !== 15 && room.type !== 17
            ).length;

            if (nonDoorRoomsCount === 0) {
                ElMessage.warning('没有可计算占比的房间');
                return;
            }

            // 计算门的总占比
            const doorAreaRate = rooms.value
                .filter(room => room.type === 15 || room.type === 17)
                .reduce((sum, room) => sum + room.areaRate, 0);

            // 剩余占比分配给非门房间
            const remainingRate = 1 - doorAreaRate;
            const ratePerRoom = remainingRate / nonDoorRoomsCount;

            rooms.value.forEach(room => {
                if (room.type !== 15 && room.type !== 17) {
                    room.areaRate = parseFloat(ratePerRoom.toFixed(2));
                }
            });

            ElMessage.success('已重新计算面积占比');
        };

        const createConnection = () => {
            if (!canCreateConnection.value) {
                ElMessage.warning('请选择两个不同的房间');
                return;
            }

            // 创建内门
            const doorIndex = rooms.value.length;
            rooms.value.push({ type: 17, corners: 4, areaRate: 0.02 });

            // 创建连接关系
            connections.value.push({
                roomA: connectionRoomA.value,
                roomB: connectionRoomB.value,
                doorIndex: doorIndex
            });

            // 重置选择
            connectionRoomA.value = null;
            connectionRoomB.value = null;

            ElMessage.success('连接关系创建成功');
        };

        const removeConnection = (index) => {
            const conn = connections.value[index];

            // 删除对应的内门
            if (conn.doorIndex < rooms.value.length) {
                rooms.value.splice(conn.doorIndex, 1);

                // 更新其他连接关系中的门索引
                connections.value.forEach(c => {
                    if (c.doorIndex > conn.doorIndex) {
                        c.doorIndex--;
                    }
                });
            }

            connections.value.splice(index, 1);
            ElMessage.success('连接关系已删除');
        };

        const handleBoundaryImageChange = (file) => {
            const isImage = file.raw.type.startsWith('image/');
            const isLt5M = file.raw.size / 1024 / 1024 < 5;

            if (!isImage) {
                ElMessage.error('只能上传图片文件');
                return;
            }

            if (!isLt5M) {
                ElMessage.error('图片大小不能超过5MB');
                return;
            }

            boundaryImage.value = file.raw;
            boundaryFileName.value = file.name;

            // 创建预览
            const reader = new FileReader();
            reader.onload = (e) => {
                boundaryImagePreview.value = e.target.result;
            };
            reader.readAsDataURL(file.raw);

            ElMessage.success('边界图上传成功');
        };

        const resetAll = () => {
            showDialog('确认重置', '确定要重置所有配置吗？此操作不可撤销。', 'warning', () => {
                rooms.value = [
                    { type: 1, corners: 4, areaRate: 0.28 },  // 客厅
                    { type: 2, corners: 4, areaRate: 0.1 },   // 厨房
                    { type: 3, corners: 4, areaRate: 0.2 },   // 卧室
                    { type: 15, corners: 4, areaRate: 0.02 }  // 前门（必须）
                ];
                connections.value = [];
                connectionRoomA.value = null;
                connectionRoomB.value = null;
                boundaryImage.value = null;
                boundaryImagePreview.value = null;
                boundaryFileName.value = '';
                generationStatus.value = 'idle';
                generationProgress.value = 0;
                generatedImageUrl.value = '';
                taskId.value = '';
                validationErrors.value = [];

                ElMessage.success('已重置所有配置');
            });
        };

        const previewJson = () => {
            console.log('previewJson called');
            console.log('generatedJson.value:', generatedJson.value);
            console.log('JSON string:', JSON.stringify(generatedJson.value, null, 2));

            // 检查formatJsonWithHighlight函数
            const highlighted = formatJsonWithHighlight(generatedJson.value);
            console.log('Highlighted HTML:', highlighted);

            // 直接设置innerHTML测试
            const jsonPreview = document.querySelector('.json-preview pre');
            if (jsonPreview) {
                jsonPreview.innerHTML = highlighted;
                console.log('直接设置innerHTML完成');
            }

            ElMessage.success('JSON已更新到预览区域');
        };

        const formatJsonWithHighlight = (json) => {
            console.log('formatJsonWithHighlight called with:', json);

            if (!json || Object.keys(json).length === 0) {
                return '<div class="json-placeholder">暂无JSON数据，请先配置房间属性</div>';
            }

            try {
                const jsonStr = JSON.stringify(json, null, 2);
                console.log('JSON string:', jsonStr);

                // 简单的语法高亮
                const highlighted = jsonStr
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?)/g, (match) => {
                        let cls = 'json-string';
                        if (/:$/.test(match)) {
                            cls = 'json-key';
                        }
                        return `<span class="${cls}">${match}</span>`;
                    })
                    .replace(/\b(true|false)\b/g, '<span class="json-boolean">$1</span>')
                    .replace(/\b(null)\b/g, '<span class="json-null">$1</span>')
                    .replace(/\b-?\d+(\.\d+)?([eE][+-]?\d+)?\b/g, '<span class="json-number">$&</span>');

                console.log('Highlighted result:', highlighted);
                return highlighted;
            } catch (error) {
                console.error('JSON格式化错误:', error);
                return `<div class="json-error">JSON格式化错误: ${error.message}</div>`;
            }
        };

        const saveJson = () => {
            console.log('saveJson called, isValid:', isValid.value);
            console.log('validationErrors:', validationErrors.value);
            console.log('generatedJson:', generatedJson.value);

            if (!isValid.value) {
                ElMessage.error({
                    message: '请先完成所有必填项并确保数据有效',
                    duration: 5000,
                    showClose: true
                });

                // 显示具体错误信息
                if (validationErrors.value.length > 0) {
                    const errorMsg = validationErrors.value.join('\n');
                    showDialog('校验失败', errorMsg, 'error');
                }
                return;
            }

            const jsonStr = JSON.stringify(generatedJson.value, null, 2);
            const blob = new Blob([jsonStr], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${generatedJson.value.name}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            ElMessage.success('JSON文件已保存');
        };

        const generateFloorPlan = async () => {
            if (!isValid.value) {
                ElMessage.error({
                    message: '请先完成所有必填项并确保数据有效',
                    duration: 5000,
                    showClose: true
                });

                // 显示具体错误信息
                if (validationErrors.value.length > 0) {
                    const errorMsg = validationErrors.value.join('\n');
                    showDialog('校验失败', errorMsg, 'error');
                }
                return;
            }

            if (!boundaryImage.value) {
                ElMessage.error('请先上传边界图');
                return;
            }

            isGenerating.value = true;
            generationStatus.value = 'generating';
            generationProgress.value = 10;
            generationMessage.value = '正在准备数据...';

            try {
                // 创建FormData
                const formData = new FormData();
                formData.append('config', JSON.stringify(generatedJson.value));
                // formData.append('boundary_image', boundaryImage.value);

                // 发送请求到后端
                const response = await axios.post('http://192.168.200.151:8000/api/generate-floor-plan', formData, {
                    headers: {
                        'Content-Type': 'multipart/form-data'
                    }
                });

                if (response.data.code === 200) {
                    taskId.value = response.data.data.task_id;
                    generationProgress.value = 30;
                    generationMessage.value = '任务已创建，开始生成平面图...';

                    // 开始轮询任务状态
                    startPollingTaskStatus();
                } else {
                    throw new Error(response.data.message);
                }
            } catch (error) {
                generationStatus.value = 'failed';
                errorMessage.value = `生成失败: ${error.message}`;
                isGenerating.value = false;
                ElMessage.error('生成失败: ' + error.message);
            }
        };

        const startPollingTaskStatus = () => {
            if (checkInterval.value) {
                clearInterval(checkInterval.value);
            }

            checkInterval.value = setInterval(async () => {
                try {
                    const response = await axios.get(`http://192.168.200.151:8000/api/task/${taskId.value}`);

                    if (response.data.code === 200) {
                        const data = response.data.data;

                        if (data.status === 'running') {
                            generationProgress.value = 50;
                            generationMessage.value = '模型推理中，预计还需 8 分钟...';
                        } else if (data.status === 'success') {
                            generationProgress.value = 100;
                            generationStatus.value = 'success';
                            generatedImageUrl.value = `http://192.168.200.151:8000${data.image_url}`;
                            generationMessage.value = '生成成功！';
                            isGenerating.value = false;
                            clearInterval(checkInterval.value);
                            ElMessage.success('平面图生成成功');
                        } else if (data.status === 'failed') {
                            generationStatus.value = 'failed';
                            errorMessage.value = `生成失败: ${response.data.message}`;
                            isGenerating.value = false;
                            clearInterval(checkInterval.value);
                            ElMessage.error('生成失败');
                        }
                    } else {
                        throw new Error(response.data.message);
                    }
                } catch (error) {
                    console.error('轮询任务状态失败:', error);
                }
            }, 3000); // 每3秒轮询一次
        };

        const cancelGeneration = async () => {
            try {
                const response = await axios.post(`http://192.168.200.151:8000/api/task/${taskId.value}/cancel`);

                if (response.data.code === 200) {
                    generationStatus.value = 'idle';
                    generationProgress.value = 0;
                    isGenerating.value = false;
                    clearInterval(checkInterval.value);
                    ElMessage.info('任务已取消');
                }
            } catch (error) {
                ElMessage.error('取消任务失败: ' + error.message);
            }
        };

        const retryGeneration = () => {
            generationStatus.value = 'idle';
            errorMessage.value = '';
            generateFloorPlan();
        };

        const downloadImage = () => {
            const a = document.createElement('a');
            a.href = generatedImageUrl.value;
            a.download = `floorplan_${taskId.value}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            ElMessage.success('图片下载成功');
        };

        const showDialog = (title, message, type = 'info', callback = null) => {
            dialogTitle.value = title;
            dialogMessage.value = message;
            dialogCallback.value = callback;
            dialogVisible.value = true;
        };

        const confirmDialog = () => {
            dialogVisible.value = false;
            if (dialogCallback.value) {
                dialogCallback.value();
            }
        };

        // 拓扑图工具提示
        const showNodeTooltip = (node) => {
            const roomTypeLabel = roomTypeOptions.find(opt => opt.value === node.roomType)?.label || `类型 ${node.roomType}`;
            const room = rooms.value[node.index];

            tooltip.value = {
                visible: true,
                x: node.x + 20,
                y: node.y - 20,
                content: `
                    <h4>房间 ${node.index}</h4>
                    <p><strong>类型:</strong> ${roomTypeLabel}</p>
                    <p><strong>拐点数量:</strong> ${room.corners}</p>
                    <p><strong>面积占比:</strong> ${(room.areaRate * 100).toFixed(1)}%</p>
                `
            };
        };

        const showDoorTooltip = (door) => {
            tooltip.value = {
                visible: true,
                x: door.x + 20,
                y: door.y - 20,
                content: `
                    <h4>内门 ${door.index}</h4>
                    <p><strong>类型:</strong> 内门 (17)</p>
                    <p><strong>拐点数量:</strong> 4</p>
                    <p><strong>面积占比:</strong> 2%</p>
                `
            };
        };

        const showLinkTooltip = (link) => {
            tooltip.value = {
                visible: true,
                x: (link.source.x + link.target.x) / 2,
                y: (link.source.y + link.target.y) / 2,
                content: `
                    <h4>连接关系</h4>
                    <p>房间 ${link.source.index} ↔ 内门 ${link.target.index}</p>
                `
            };
        };

        const hideTooltip = () => {
            tooltip.value.visible = false;
        };

        // 监听窗口大小变化，调整拓扑图尺寸
        onMounted(() => {
            const updateTopologySize = () => {
                const container = document.querySelector('.topology-container');
                if (container) {
                    topologyWidth.value = container.clientWidth;
                    topologyHeight.value = container.clientHeight;
                }
            };

            updateTopologySize();
            window.addEventListener('resize', updateTopologySize);
        });

        // 监听数据变化，自动更新JSON预览
        watch([rooms, connections, boundaryFileName], () => {
            // JSON预览会自动更新
        }, { deep: true });

        return {
            // 数据
            rooms,
            connections,
            connectionRoomA,
            connectionRoomB,
            boundaryImagePreview,
            boundaryFileName,
            dialogVisible,
            dialogTitle,
            dialogMessage,
            generationStatus,
            generationProgress,
            generationMessage,
            generatedImageUrl,
            errorMessage,
            isGenerating,
            validationErrors,
            jsonText,
            jsonError,

            // 计算属性
            roomTypeOptions,
            hasFrontDoor,
            nonDoorRooms,
            canCreateConnection,
            isValid,
            generatedJson,
            formattedJson,
            topologyNodes,
            topologyDoors,
            topologyLinks,
            topologyWidth,
            topologyHeight,
            tooltip,

            // 方法
            addRoom,
            removeRoom,
            setAsFrontDoor,
            calculateProportions,
            createConnection,
            removeConnection,
            handleBoundaryImageChange,
            resetAll,
            previewJson,
            saveJson,
            generateFloorPlan,
            cancelGeneration,
            retryGeneration,
            downloadImage,
            confirmDialog,
            showNodeTooltip,
            showDoorTooltip,
            showLinkTooltip,
            hideTooltip,
            formatJsonWithHighlight,
            onJsonInput,
            formatJson,
            updateFromJson
        };
    }
}).use(ElementPlus).mount('#app');
