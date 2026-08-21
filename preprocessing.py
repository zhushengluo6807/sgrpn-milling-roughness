"""
铣削振动信号预处理模块
========================
基于"第二章物理失效改进.docx"需求，对112组实验振动数据进行预处理。

预处理流水线（6步）：
  1. 数据加载与解析
  2. 固定频率干扰陷波（50Hz工频 + 系统谐振点）
  3. 异常跳变检测与修复（Hampel滤波器）
  4. 直流偏置去除（中值去趋势）
  5. 切削段自适应检测（双阈值能量法）
  6. 切削段截取与有效性标记

注意：本模块不包含滑动窗口划分（暂不使用）。
      预处理后的信号将作为1D-CNN特征提取模块的输入。

作者：自动生成
日期：2026-06-08
"""

import sys
import os
# 强制UTF-8编码，解决Windows GBK控制台乱码问题
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import median_filter
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 全局配置
# ============================================================
CONFIG = {
    "fs": 25600,                    # 采样频率 (Hz)
    "total_duration": 30.0,         # 每条记录总时长 (s)
    "channels": ["ch9", "ch10", "ch11"],  # 三轴振动通道
    "channel_labels": ["X [s]", "Real [g]", "Real [g]", "Real [g]"],

    # Step 2: 陷波滤波器配置
    "notch_frequencies": {
        "powerline": [50, 100, 150, 200, 250, 300, 350],  # 工频及其谐波 (Hz)
        "system_resonance": [4000, 6000, 8000, 8890],       # 系统谐振点 (Hz)
    },
    "notch_q": 30,                   # 品质因数（越高越窄）

    # Step 3: Hampel滤波器配置
    "hampel_window": 256,            # 滑动窗口大小（约10ms @ 25.6kHz）
    "hampel_n_sigma": 5,             # MAD倍数阈值（保守，避免误伤真实切削信号）

    # Step 5: 切削段检测配置
    "energy_window": 0.02,           # 短时能量计算窗口 (s)
    "energy_high_ratio": 8.0,        # 高阈值 = 噪声基准 × ratio（进刀检测）
    "energy_low_ratio": 3.0,         # 低阈值 = 噪声基准 × ratio（退刀检测）
    "min_cutting_duration": 0.05,    # 最短有效切削段 (s)
    "max_gap_merge": 0.5,            # 合并此间隔内的相邻切削段 (s)
    "margin_before": 0.1,            # 切削段前扩展裕量 (s)
    "margin_after": 0.1,             # 切削段后扩展裕量 (s)

    # Step 6: 输出配置
    "output_dir": "预处理结果",
}

# ============================================================
# 实验记录参数映射（从实验记录表.xlsx提取）
# 格式：{执行顺序: (块号, 层, 槽号, 转速rpm, 进给mm/z, 切深mm)}
# ============================================================
EXPERIMENT_PARAMS = {
    1:  (1,1,1, 5000,0.10,0.8),  2:  (1,1,2, 4000,0.04,0.8),
    3:  (1,1,3, 7000,0.10,0.5),  4:  (1,1,4, 4000,0.10,1.0),
    5:  (1,1,5, 7000,0.06,0.5),  6:  (1,1,6, 4000,0.02,0.8),
    7:  (1,1,7, 6000,0.02,0.8),  8:  (1,1,8, 8000,0.08,0.2),
    9:  (1,2,1, 5000,0.02,0.5),  10: (1,2,2, 4000,0.06,0.2),
    11: (1,2,3, 7000,0.02,1.0),  12: (1,2,4, 6000,0.06,0.8),
    13: (1,2,5, 8000,0.08,0.5),  14: (1,2,6, 8000,0.02,0.2),
    15: (1,2,7, 5000,0.02,0.8),  16: (1,2,8, 5000,0.02,0.2),
    17: (2,1,1, 8000,0.06,1.0),  18: (2,1,2, 4000,0.10,0.5),
    19: (2,1,3, 6000,0.04,1.0),  20: (2,1,4, 8000,0.10,1.0),
    21: (2,1,5, 6000,0.06,0.8),  22: (2,1,6, 8000,0.08,0.8),
    23: (2,1,7, 6000,0.06,1.0),  24: (2,1,8, 4000,0.02,1.0),
    25: (2,2,1, 6000,0.04,0.8),  26: (2,2,2, 8000,0.06,0.8),
    27: (2,2,3, 4000,0.04,0.2),  28: (2,2,4, 5000,0.04,0.2),
    29: (2,2,5, 5000,0.10,1.0),  30: (2,2,6, 8000,0.02,0.2),
    31: (2,2,7, 6000,0.10,1.0),  32: (3,2,5, 4000,0.04,1.0),
    33: (3,1,1, 6000,0.08,1.0),  34: (3,1,2, 6000,0.08,0.2),
    35: (3,1,3, 7000,0.02,0.5),  36: (3,1,4, 8000,0.02,1.0),
    37: (3,1,5, 7000,0.08,0.2),  38: (3,1,6, 6000,0.06,0.8),
    39: (3,1,7, 4000,0.10,0.2),  40: (3,1,8, 5000,0.08,0.5),
    41: (3,2,1, 8000,0.10,0.5),  42: (3,2,2, 8000,0.06,0.2),
    43: (3,2,3, 8000,0.04,1.0),  44: (3,2,4, 6000,0.06,0.5),
    45: (2,2,8, 8000,0.08,1.0),  46: (3,2,6, 5000,0.06,1.0),
    47: (3,2,7, 8000,0.02,0.2),  48: (3,2,8, 7000,0.08,0.8),
    49: (4,1,1, 5000,0.10,0.5),  50: (4,1,2, 5000,0.04,0.5),
    51: (4,1,3, 4000,0.04,0.5),  52: (4,1,4, 5000,0.04,1.0),
    53: (4,1,5, 6000,0.10,0.8),  54: (4,1,6, 7000,0.04,1.0),
    55: (4,1,7, 5000,0.02,1.0),  56: (4,1,8, 4000,0.10,1.0),
    57: (4,2,1, 8000,0.06,0.5),  58: (4,2,2, 8000,0.04,0.5),
    59: (4,2,3, 4000,0.10,1.0),  60: (4,2,4, 8000,0.02,0.5),
    61: (4,2,5, 6000,0.08,0.8),  62: (4,2,6, 7000,0.06,0.2),
    63: (4,2,7, 6000,0.02,0.5),  64: (4,2,8, 5000,0.04,0.5),
    65: (5,1,1, 8000,0.04,0.2),  66: (5,1,2, 7000,0.04,0.8),
    67: (5,1,3, 5000,0.06,0.5),  68: (5,1,4, 7000,0.02,0.2),
    69: (5,1,5, 4000,0.10,0.8),  70: (5,1,6, 5000,0.08,0.8),
    71: (5,1,7, 5000,0.08,1.0),  72: (5,1,8, 4000,0.06,1.0),
    73: (5,2,1, 5000,0.08,1.0),  74: (5,2,2, 6000,0.02,1.0),
    75: (5,2,3, 4000,0.06,0.5),  76: (5,2,4, 8000,0.10,0.2),
    77: (5,2,5, 6000,0.06,0.2),  78: (5,2,6, 6000,0.04,0.5),
    79: (5,2,7, 6000,0.04,0.2),  80: (5,2,8, 5000,0.08,0.2),
    81: (6,1,1, 6000,0.02,1.0),  82: (6,1,2, 6000,0.06,0.8),
    83: (6,1,3, 5000,0.04,0.8),  84: (6,1,4, 8000,0.10,0.8),
    85: (6,1,5, 8000,0.04,0.8),  86: (6,1,6, 4000,0.02,0.5),
    87: (6,1,7, 6000,0.10,0.5),  88: (6,1,8, 4000,0.08,0.2),
    89: (6,2,1, 5000,0.10,0.2),  90: (6,2,2, 7000,0.10,0.5),
    91: (6,2,3, 4000,0.08,0.8),  92: (6,2,4, 4000,0.06,0.8),
    93: (6,2,5, 7000,0.10,0.8),  94: (6,2,6, 4000,0.02,0.2),
    95: (6,2,7, 5000,0.06,0.8),  96: (6,2,8, 6000,0.10,0.2),
    97: (7,1,1, 7000,0.08,1.0),  98: (7,1,2, 7000,0.02,0.8),
    99: (7,1,3, 8000,0.06,0.8),  100:(7,1,4, 7000,0.06,0.8),
    101:(7,1,5, 7000,0.08,0.5),  102:(7,1,6, 6000,0.08,0.5),
    103:(7,1,7, 7000,0.06,1.0),  104:(7,1,8, 7000,0.10,1.0),
    105:(7,2,1, 7000,0.10,0.2),  106:(7,2,2, 5000,0.06,0.2),
    107:(7,2,3, 4000,0.08,0.5),  108:(7,2,4, 8000,0.02,0.8),
    109:(7,2,5, 7000,0.04,0.5),  110:(7,2,6, 7000,0.04,0.2),
    111:(7,2,7, 6000,0.02,0.2),  112:(7,2,8, 4000,0.08,1.0),
}
# 重复实验标记 (R01~R12)
REPEAT_MARKERS = {4:"R05", 12:"R02", 21:"R03", 30:"R07", 38:"R01",
                  47:"R06", 56:"R04", 64:"R08", 73:"R09", 81:"R10",
                  90:"R11", 99:"R12"}


# ============================================================
# Step 1: 数据加载与解析
# ============================================================
def load_data(filepath: str) -> dict:
    """
    加载单个.txt振动数据文件。

    文件格式（制表符分隔）：
        行1: 列标题 "Time record(Channel 9)\tTime record(Channel 10)\t..."
        行2: 单位标题 "X [s]\tReal [g]\tReal [g]\tReal [g]"
        行3+: 数据行（科学计数法）

    Returns:
        dict: {
            "time": 时间数组 (s),
            "ch9": 通道9振动 (g),
            "ch10": 通道10振动 (g),
            "ch11": 通道11振动 (g),
            "fs": 采样频率,
            "n_samples": 采样点数,
        }
    """
    data = np.loadtxt(filepath, skiprows=2)
    return {
        "time": data[:, 0],
        "ch9": data[:, 1],
        "ch10": data[:, 2],
        "ch11": data[:, 3],
        "fs": CONFIG["fs"],
        "n_samples": len(data),
    }


# ============================================================
# Step 2: 固定频率干扰陷波
# ============================================================
def apply_notch_filters(data: dict) -> dict:
    """
    使用IIR陷波滤波器去除已知的固定频率干扰。

    去除频点：
      - 工频干扰：50Hz 及其谐波（100/150/200/250/300/350 Hz）
      - 系统谐振：4000/6000/8000/8890 Hz
      （数据采集中发现的固定干扰——非工况相关，不影响刀齿通过频率信息）

    设计依据（第二章§2.2）：
      1D-CNN的卷积核本质上是可学习的带通滤波器组。预处理阶段去除确知的
      固定干扰可减少CNN对无关频率的学习负担，使其更专注于对粗糙度残差
      有意义的铣削颤振频率、刀齿通过频率边带等动态成分。
    """
    fs = data["fs"]
    result = data.copy()

    # 合并所有陷波频率
    all_notch_freqs = (
        CONFIG["notch_frequencies"]["powerline"] +
        CONFIG["notch_frequencies"]["system_resonance"]
    )

    for freq in all_notch_freqs:
        if freq >= fs / 2:  # 超过奈奎斯特频率，跳过
            continue
        # 设计二阶IIR陷波滤波器
        b, a = signal.iirnotch(freq, CONFIG["notch_q"], fs)
        for ch in CONFIG["channels"]:
            result[ch] = signal.filtfilt(b, a, result[ch])

    return result


# ============================================================
# Step 3: 异常跳变检测与修复（Hampel滤波器）
# ============================================================
def remove_spikes_hampel(data: dict) -> dict:
    """
    使用Hampel辨识法检测并替换异常跳变尖峰。

    方法：
      对每个采样点，在其局部窗口中计算中位数和MAD（中位数绝对偏差）。
      若 |x_i - median| > n_sigma * MAD，判定为离群点。
      离群点用该窗口的中位数替代。

    参数选择依据：
      - 窗口256点 ≈ 10ms：足够捕获局部统计特性，又不会被切削瞬态淹没
      - n_sigma=5：保守阈值。真实切削冲击（如刀齿切入）的幅值可达数g，
        过小阈值会误伤切削信号本身。5倍MAD足以过滤传感器跳变（通常
        表现为单点或极短脉冲）同时保留真实的切削动力学特征。

    注意：
      此步骤在陷波之后、去直流之前执行，因为MAD计算需要信号的原始分布。
    """
    result = data.copy()
    window = CONFIG["hampel_window"]
    n_sigma = CONFIG["hampel_n_sigma"]

    for ch in CONFIG["channels"]:
        x = result[ch]
        # 使用scipy的median_filter高效计算滑动中位数
        rolling_median = median_filter(x, size=window, mode="reflect")
        # MAD = median(|x - median|)
        abs_deviation = np.abs(x - rolling_median)
        rolling_mad = median_filter(abs_deviation, size=window, mode="reflect")
        # 防止MAD=0导致除零
        rolling_mad = np.maximum(rolling_mad, 1e-10)

        # 检测离群点
        z_score = np.abs(x - rolling_median) / rolling_mad
        outliers = z_score > n_sigma

        n_outliers = np.sum(outliers)
        if n_outliers > 0:
            # 用局部中位数替换离群点
            x_clean = x.copy()
            x_clean[outliers] = rolling_median[outliers]
            result[ch] = x_clean

    return result


# ============================================================
# Step 4: 直流偏置去除
# ============================================================
def remove_dc_offset(data: dict) -> dict:
    """
    去除各通道的直流偏置。

    方法：每通道减去其中位数（比均值更鲁棒，不受离群点影响）。

    设计依据：
      加速度计在静止时应输出0g。实际测量中的直流偏置可能来自：
        - 传感器零漂
        - 采集卡偏置电压
        - 重力分量（传感器安装姿态）
      去除直流后，1D-CNN可以专注于信号的动态（AC）成分。
      中位数减法保留了信号的波形形态，优于线性去趋势（后者可能
      扭曲短时切削瞬态的低频包络）。
    """
    result = data.copy()
    for ch in CONFIG["channels"]:
        result[ch] = result[ch] - np.median(result[ch])
    return result


# ============================================================
# Step 5: 切削段自适应检测
# ============================================================
def detect_cutting_segment(data: dict) -> dict:
    """
    自适应切削段检测 —— 双模式策略。

    数据中存在两种采集模式：
      - 模式1（双模态）: 信号包含明确的"安静→切削→安静"过渡。
                      能量分布呈双峰结构。适用于低噪/中噪文件。
                      使用双阈值VAD检测切削段边界。
      - 模式2（连续切削）: 全程背景能量较高，无明显安静段。
                      可能为连续走刀或主轴全程开启的连续采集。
                      此时将整段信号视为切削数据，仅裁剪首尾
                      可能存在的启动/停止瞬态（各0.5s）。

    分类依据：
      计算能量中位数 p50 与能量10%分位 p10 的比值。
      - 若 p50/p10 > 3：存在明显安静段 → 模式1（双阈值VAD）
      - 若 p50/p10 ≤ 3 且 p10 > 0.003g：全程高能 → 模式2（连续切削）
      - 若 p50/p10 ≤ 3 且 p10 ≤ 0.003g：全程安静 → 无切削，标记无效
    """
    fs = data["fs"]
    result = data.copy()

    # 使用三通道合成的RMS能量（更稳健）
    combined = np.sqrt(
        data["ch9"]**2 + data["ch10"]**2 + data["ch11"]**2
    ) / np.sqrt(3)

    # 短时RMS能量
    win_samples = int(CONFIG["energy_window"] * fs)
    n_windows = len(combined) // win_samples
    energy = np.array([
        np.sqrt(np.mean(combined[i*win_samples:(i+1)*win_samples]**2))
        for i in range(n_windows)
    ])

    # 能量分布统计
    p10 = np.percentile(energy, 10)
    p50 = np.percentile(energy, 50)
    p90 = np.percentile(energy, 90)
    max_energy = np.max(energy)
    burst_ratio = max_energy / max(p10, 1e-10)  # 爆冲比：最大能量 vs 安静基准

    # === 模式判定（基于 max/p10 比值，而非 p50/p10） ===
    # 原因：低噪文件中切削段仅占0.1~5s，p50仍反映安静段水平。
    #      使用 max/p10 可正确识别短暂但强烈的切削脉冲。
    if burst_ratio > 50 and p10 < 0.005:
        # 模式1：双模态（有明显安静段 + 强切削段）
        # 例：文件1, max/p10=151x, p10=0.0005g
        mode = "bimodal"
        noise_floor = p10
        high_thresh = noise_floor * CONFIG["energy_high_ratio"]
        low_thresh = noise_floor * CONFIG["energy_low_ratio"]

        # 双阈值滞回检测
        in_cutting = False
        segments = []
        seg_start = None

        for i in range(n_windows):
            if not in_cutting:
                if energy[i] > high_thresh:
                    in_cutting = True
                    seg_start = i
            else:
                if energy[i] < low_thresh:
                    in_cutting = False
                    if seg_start is not None:
                        segments.append((seg_start, i))
                        seg_start = None
        if in_cutting and seg_start is not None:
            segments.append((seg_start, n_windows - 1))

        # 转换为时间并后处理
        segments_time = []
        for s, e in segments:
            t_start = s * CONFIG["energy_window"]
            t_end = (e + 1) * CONFIG["energy_window"]
            segments_time.append((t_start, t_end))

        # 合并相邻段
        if len(segments_time) > 1:
            merged = [segments_time[0]]
            for seg in segments_time[1:]:
                if seg[0] - merged[-1][1] < CONFIG["max_gap_merge"]:
                    merged[-1] = (merged[-1][0], seg[1])
                else:
                    merged.append(seg)
            segments_time = merged

        # 过滤过短段
        segments_time = [
            seg for seg in segments_time
            if seg[1] - seg[0] >= CONFIG["min_cutting_duration"]
        ]

        if len(segments_time) == 0:
            result["cutting_segment"] = None
            result["is_valid"] = False
            result["detection_info"] = {
                "mode": mode,
                "p10": p10, "p50": p50, "p90": p90,
                "burst_ratio": burst_ratio,
                "noise_floor": noise_floor,
                "high_thresh": high_thresh, "low_thresh": low_thresh,
                "segments_found": 0,
                "warning": "双模态模式：未检测到切削段——可能未采集到进刀/退刀时刻",
            }
        else:
            main_seg = max(segments_time, key=lambda s: s[1] - s[0])
            t_start = max(0, main_seg[0] - CONFIG["margin_before"])
            t_end = min(CONFIG["total_duration"], main_seg[1] + CONFIG["margin_after"])
            result["cutting_segment"] = (t_start, t_end)
            result["is_valid"] = True
            result["detection_info"] = {
                "mode": mode,
                "p10": p10, "p50": p50, "p90": p90,
                "burst_ratio": burst_ratio,
                "noise_floor": noise_floor,
                "high_thresh": high_thresh, "low_thresh": low_thresh,
                "segments_found": len(segments_time),
                "all_segments": segments_time,
                "main_segment": main_seg,
                "segment_with_margin": (t_start, t_end),
            }

    elif burst_ratio > 10 and p10 < 0.005:
        # 模式1b：弱双模态（安静段 + 较弱切削段）
        # 仍尝试双阈值检测，但使用更低的阈值倍数
        mode = "bimodal_weak"
        noise_floor = p10
        high_thresh = noise_floor * (CONFIG["energy_high_ratio"] * 0.5)
        low_thresh = noise_floor * (CONFIG["energy_low_ratio"] * 0.5)

        in_cutting = False
        segments = []
        seg_start = None

        for i in range(n_windows):
            if not in_cutting:
                if energy[i] > high_thresh:
                    in_cutting = True
                    seg_start = i
            else:
                if energy[i] < low_thresh:
                    in_cutting = False
                    if seg_start is not None:
                        segments.append((seg_start, i))
                        seg_start = None
        if in_cutting and seg_start is not None:
            segments.append((seg_start, n_windows - 1))

        segments_time = []
        for s, e in segments:
            t_start = s * CONFIG["energy_window"]
            t_end = (e + 1) * CONFIG["energy_window"]
            segments_time.append((t_start, t_end))

        if len(segments_time) > 1:
            merged = [segments_time[0]]
            for seg in segments_time[1:]:
                if seg[0] - merged[-1][1] < CONFIG["max_gap_merge"]:
                    merged[-1] = (merged[-1][0], seg[1])
                else:
                    merged.append(seg)
            segments_time = merged

        segments_time = [
            seg for seg in segments_time
            if seg[1] - seg[0] >= CONFIG["min_cutting_duration"]
        ]

        if len(segments_time) == 0:
            result["cutting_segment"] = None
            result["is_valid"] = False
            result["detection_info"] = {
                "mode": mode,
                "p10": p10, "p50": p50, "p90": p90,
                "burst_ratio": burst_ratio,
                "noise_floor": noise_floor,
                "high_thresh": high_thresh, "low_thresh": low_thresh,
                "segments_found": 0,
                "warning": "弱双模态模式：切削段能量偏低，未检测到有效切削段",
            }
        else:
            main_seg = max(segments_time, key=lambda s: s[1] - s[0])
            t_start = max(0, main_seg[0] - CONFIG["margin_before"])
            t_end = min(CONFIG["total_duration"], main_seg[1] + CONFIG["margin_after"])
            result["cutting_segment"] = (t_start, t_end)
            result["is_valid"] = True
            result["detection_info"] = {
                "mode": mode,
                "p10": p10, "p50": p50, "p90": p90,
                "burst_ratio": burst_ratio,
                "noise_floor": noise_floor,
                "high_thresh": high_thresh, "low_thresh": low_thresh,
                "segments_found": len(segments_time),
                "all_segments": segments_time,
                "main_segment": main_seg,
                "segment_with_margin": (t_start, t_end),
            }

    elif p10 > 0.003:
        # 模式2：连续切削（全程高能，无明显安静段）
        mode = "continuous"
        # 裁剪首尾可能的瞬态（各0.5s）
        trim_start = 0.5
        trim_end = 0.5
        t_start = max(0, trim_start)
        t_end = max(t_start + 0.1, CONFIG["total_duration"] - trim_end)

        result["cutting_segment"] = (t_start, t_end)
        result["is_valid"] = True
        result["detection_info"] = {
            "mode": mode,
            "p10": p10, "p50": p50, "p90": p90,
            "burst_ratio": burst_ratio,
            "noise_floor": p10,
            "warning": "连续切削模式：全程高振动能量，视为连续切削过程，已裁剪首尾0.5s瞬态",
            "trim_start": trim_start, "trim_end": trim_end,
        }

    else:
        # 无效：全程能量极低，可能传感器未启动或无切削
        mode = "silent"
        result["cutting_segment"] = None
        result["is_valid"] = False
        result["detection_info"] = {
            "mode": mode,
            "p10": p10, "p50": p50, "p90": p90,
            "burst_ratio": burst_ratio,
            "warning": "全程安静模式：能量极低，可能传感器未正常工作或无切削发生",
        }

    return result


# ============================================================
# Step 6: 切削段截取
# ============================================================
def extract_cutting_signal(data: dict) -> dict:
    """
    根据检测到的切削段时间范围，截取原始信号。

    输出：
      - cutting_time: 切削段时间轴 (重设从0开始)
      - cutting_ch9/10/11: 切削段振动信号
      - pre_cut_noise: 切削前的噪声段（用于信噪比评估）
    """
    result = data.copy()
    seg = data.get("cutting_segment")

    if seg is None:
        return result

    fs = data["fs"]
    idx_start = int(seg[0] * fs)
    idx_end = int(seg[1] * fs)

    # 确保边界有效
    idx_start = max(0, min(idx_start, data["n_samples"] - 1))
    idx_end = max(idx_start + 1, min(idx_end, data["n_samples"]))

    # 截取切削段
    result["cutting_time"] = data["time"][idx_start:idx_end] - seg[0]
    result["cutting_ch9"] = data["ch9"][idx_start:idx_end]
    result["cutting_ch10"] = data["ch10"][idx_start:idx_end]
    result["cutting_ch11"] = data["ch11"][idx_start:idx_end]

    # 切削前噪声段（前2秒，若存在）
    noise_end = min(idx_start, 2 * fs)
    if noise_end > 0:
        noise_start = max(0, idx_start - noise_end)
        result["pre_cut_noise_ch9"] = data["ch9"][noise_start:idx_start]
        result["pre_cut_noise_rms"] = np.sqrt(np.mean(
            data["ch9"][noise_start:idx_start]**2
        ))
    else:
        result["pre_cut_noise_ch9"] = np.array([])
        result["pre_cut_noise_rms"] = 0.0

    # 计算切削段信噪比
    cutting_rms = np.sqrt(np.mean(result["cutting_ch9"]**2))
    if result["pre_cut_noise_rms"] > 0:
        result["snr_db"] = 20 * np.log10(cutting_rms / result["pre_cut_noise_rms"])
    else:
        result["snr_db"] = float("inf")

    return result


# ============================================================
# 主预处理流水线
# ============================================================
def preprocess_single(filepath: str, order: int) -> dict:
    """
    对单个数据文件执行完整预处理流水线。

    流水线顺序（关键——先检测后清洗）：
      Step 1: 加载原始数据
      Step 2: 轻量去直流（仅用于切削段检测，不改变原信号）
      Step 3: 切削段检测（在原始信号能量基础上 —— 避免陷波滤除固定
              频率后能量骤降导致误判为"无声"）
      Step 4: 切削段截取（从原始信号中截取）
      Step 5: 对切削段施加完整清洗流水线：
              a. 陷波滤波（去除50Hz工频、系统谐振）
              b. Hampel去尖峰（异常跳变修复）
              c. 中值去直流（最终零均值化）
      Step 6: 保存清洁切削段数据

    设计理由：
      陷波滤波器会去除~8890Hz等传感器谐振成分，这些成分在低噪型文件
      （A类）中贡献了大部分RMS能量。若在切削检测前进行陷波，低噪文件
      的能量会降至与安静段难以区分的水平，导致全部误判为"无声"。
      因此先利用原始信号的丰富能量进行检测，再对已截取的切削段做精细清洗。
    """
    params = EXPERIMENT_PARAMS.get(order, (0, 0, 0, 0, 0, 0))
    block, layer, slot, rpm, fz, ap = params

    result = {
        "file": str(filepath),
        "order": order,
        "block": block,
        "layer": layer,
        "slot": slot,
        "rpm": rpm,
        "feed_per_tooth": fz,
        "depth_of_cut": ap,
        "tpf_hz": rpm * 3 / 60,
        "is_repeat": order in REPEAT_MARKERS,
        "repeat_label": REPEAT_MARKERS.get(order, ""),
        "preprocessing_steps": {},
    }

    # === Step 1: 加载原始数据 ===
    data = load_data(filepath)
    result["n_samples_raw"] = data["n_samples"]

    # === Step 2: 轻量去直流（仅用于切削段检测） ===
    # 创建一份副本用于检测，不修改原始数据
    data_for_detection = {
        "time": data["time"],
        "ch9": data["ch9"] - np.median(data["ch9"]),
        "ch10": data["ch10"] - np.median(data["ch10"]),
        "ch11": data["ch11"] - np.median(data["ch11"]),
        "fs": data["fs"],
        "n_samples": data["n_samples"],
    }

    # === Step 3: 切削段检测（使用去直流后的信号） ===
    data_for_detection = detect_cutting_segment(data_for_detection)
    result["is_valid"] = data_for_detection["is_valid"]
    result["detection_info"] = data_for_detection.get("detection_info", {})

    # === Step 4: 从原始信号中截取切削段 ===
    # 将检测结果映射回原始数据
    cutting_seg = data_for_detection.get("cutting_segment")
    if cutting_seg is not None:
        fs = data["fs"]
        idx_start = int(cutting_seg[0] * fs)
        idx_end = int(cutting_seg[1] * fs)
        idx_start = max(0, min(idx_start, data["n_samples"] - 1))
        idx_end = max(idx_start + 1, min(idx_end, data["n_samples"]))

        # 截取原始切削段
        cutting_raw = {
            "time": data["time"][idx_start:idx_end] - cutting_seg[0],
            "ch9": data["ch9"][idx_start:idx_end],
            "ch10": data["ch10"][idx_start:idx_end],
            "ch11": data["ch11"][idx_start:idx_end],
            "fs": fs,
            "n_samples": idx_end - idx_start,
        }
    else:
        cutting_raw = None

    # === Step 5: 对切削段施加完整清洗流水线 ===
    if cutting_raw is not None:
        # 5a: 陷波滤波
        cutting_clean = apply_notch_filters(cutting_raw)
        result["preprocessing_steps"]["notch_filtered"] = True

        # 5b: 去尖峰
        cutting_clean = remove_spikes_hampel(cutting_clean)
        result["preprocessing_steps"]["despiked"] = True

        # 5c: 最终去直流
        cutting_clean = remove_dc_offset(cutting_clean)
        result["preprocessing_steps"]["dc_removed"] = True

        # 保存清洁切削段
        result["cutting_duration"] = cutting_seg[1] - cutting_seg[0]
        result["cutting_data"] = {
            "time": cutting_clean["time"],
            "ch9": cutting_clean["ch9"],
            "ch10": cutting_clean["ch10"],
            "ch11": cutting_clean["ch11"],
        }

        # 信噪比：切削段RMS / 切削前背景RMS
        # 背景取自切削前的非切削部分（从原始数据中取）
        if idx_start > 0:
            pre_cut_raw = data["ch9"][max(0, idx_start - int(2*fs)):idx_start]
            if len(pre_cut_raw) > 0:
                noise_rms = np.sqrt(np.mean((pre_cut_raw - np.median(pre_cut_raw))**2))
                signal_rms = np.sqrt(np.mean(cutting_clean["ch9"]**2))
                if noise_rms > 0:
                    result["snr_db"] = 20 * np.log10(signal_rms / max(noise_rms, 1e-10))
                else:
                    result["snr_db"] = float("inf")
            else:
                result["snr_db"] = None
        else:
            result["snr_db"] = None
    else:
        result["cutting_data"] = None
        result["snr_db"] = None

    # 保留原始全段信号（供参考，不用于训练）
    result["full_signal_raw"] = {
        "time": data["time"],
        "ch9": data["ch9"],
        "ch10": data["ch10"],
        "ch11": data["ch11"],
    }

    return result


def preprocess_all(data_dir: str, output_dir: str):
    """
    批量预处理全部112组实验数据。

    Args:
        data_dir: 实验数据文件夹路径
        output_dir: 预处理结果输出目录
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = []
    valid_count = 0
    invalid_orders = []

    print(f"{'='*60}")
    print(f"铣削振动信号预处理")
    print(f"数据目录: {data_path}")
    print(f"输出目录: {output_path}")
    print(f"{'='*60}\n")

    for order in range(1, 113):
        filepath = data_path / f"{order}.txt"
        if not filepath.exists():
            print(f"[警告] 执行序{order}: 文件 {order}.txt 不存在，跳过")
            continue

        print(f"[{order:3d}/112] 处理中...", end=" ")

        try:
            result = preprocess_single(str(filepath), order)
            all_results.append(result)

            if result["is_valid"]:
                valid_count += 1
                snr_str = f"{result['snr_db']:.1f}dB" if result.get('snr_db') is not None else "N/A"
                print(f"块{result['block']}-L{result['layer']}-S{result['slot']} "
                      f"| n={result['rpm']}rpm fz={result['feed_per_tooth']} ap={result['depth_of_cut']} "
                      f"| 切削段={result['cutting_duration']:.2f}s SNR={snr_str}")
            else:
                invalid_orders.append(order)
                print(f"⚠ 无效（{result['detection_info'].get('warning', '未检出切削段')}）")

        except Exception as e:
            print(f"✗ 错误: {e}")
            invalid_orders.append(order)

    # 保存结果
    # 1. 每个文件的切削段数据（.npz格式）
    cutting_dir = output_path / "cutting_signals"
    cutting_dir.mkdir(exist_ok=True)
    for r in all_results:
        if r["is_valid"] and r["cutting_data"] is not None:
            np.savez_compressed(
                cutting_dir / f"order_{r['order']:03d}.npz",
                time=r["cutting_data"]["time"],
                ch9=r["cutting_data"]["ch9"],
                ch10=r["cutting_data"]["ch10"],
                ch11=r["cutting_data"]["ch11"],
                order=r["order"],
                rpm=r["rpm"],
                feed=r["feed_per_tooth"],
                doc=r["depth_of_cut"],
            )

    # 2. 汇总报告（.json）
    summary = {
        "total_files": 112,
        "processed": len(all_results),
        "valid": valid_count,
        "invalid": len(invalid_orders),
        "invalid_orders": invalid_orders,
        "config": CONFIG,
    }
    with open(output_path / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    # 3. 详细结果表（.csv）
    rows = []
    for r in all_results:
        rows.append({
            "order": r["order"],
            "block": r["block"],
            "layer": r["layer"],
            "slot": r["slot"],
            "rpm": r["rpm"],
            "feed_mm_z": r["feed_per_tooth"],
            "doc_mm": r["depth_of_cut"],
            "tpf_hz": r["tpf_hz"],
            "is_repeat": r["is_repeat"],
            "is_valid": r["is_valid"],
            "cutting_duration_s": r.get("cutting_duration"),
            "snr_db": r.get("snr_db"),
            "n_samples_raw": r["n_samples_raw"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path / "preprocessing_results.csv", index=False, encoding="utf-8-sig")

    # 打印汇总
    print(f"\n{'='*60}")
    print(f"预处理完成")
    print(f"  总文件数: 112")
    print(f"  有效切削段: {valid_count}")
    print(f"  无效/未检出: {len(invalid_orders)}")
    if invalid_orders:
        print(f"  无效执行序: {invalid_orders}")
    print(f"  结果保存至: {output_path}")
    print(f"{'='*60}")

    return all_results, summary


# ============================================================
# 命令行入口
# ============================================================
if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "实验数据"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else CONFIG["output_dir"]

    preprocess_all(data_dir, output_dir)
