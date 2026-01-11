"""
================================================================================
栅格数据NoData填充工具 (Raster NoData Filling Toolkit)
================================================================================

功能描述:
    为栅格数据（GeoTIFF等格式）提供灵活的NoData值填充功能。
    支持InSAR形变数据、DEM、多光谱/RGB图像等各类地理空间数据。

主要特性:
    ✓ 多种填充策略：均值填充、0填充、自定义值填充
    ✓ 波段独立处理：支持按波段分别处理或全局统一处理
    ✓ 批量处理：一键处理整个目录的栅格文件
    ✓ 便捷函数：提供快捷方式，简化常用操作
    ✓ 完整日志：详细的处理过程输出，便于追踪

依赖库:
    - numpy: 数值计算
    - rasterio: 栅格数据读写（基于GDAL）
    - os: 文件系统操作


快速使用示例:
-------------
>>> # 示例1: 使用0填充DEM数据（便捷方式）
>>> process_raster_with_zero("dem.tif", "dem_filled.tif")

>>> # 示例2: 使用波段均值填充RGB图像（默认行为）
>>> process_raster("rgb.tif", "rgb_filled.tif")

>>> # 示例3: 使用全局均值填充InSAR数据
>>> process_raster("insar.tif", "insar_filled.tif", fill_method="global_mean")

>>> # 示例4: 使用自定义值填充
>>> process_raster("data.tif", "data_filled.tif", fill_value=-9999)

>>> # 示例5: 批量处理
>>> batch_process("input_dir/", "output_dir/", fill_value=0)

详细用法请参考main函数中的示例代码。
================================================================================
"""

import numpy as np
import rasterio
import os
import glob


# ============================================================================
# 内部辅助函数 (Internal Helper Functions)
# ============================================================================


def _create_nodata_mask(band_data, nodata_value):
    """
    创建NoData掩码（内部辅助函数）。

    正确处理None和NaN类型的NoData值，返回布尔掩码数组。

    参数:
    ----------
    band_data : numpy.ndarray
        波段数据数组
    nodata_value : numeric or None or NaN
        NoData值

    返回:
    -------
    mask : numpy.ndarray (bool)
        布尔掩码，True表示NoData位置
    """
    if nodata_value is None or (isinstance(nodata_value, float) and np.isnan(nodata_value)):
        return np.isnan(band_data)
    else:
        return band_data == nodata_value


# ============================================================================
# 核心处理函数 (Core Processing Functions)
# ============================================================================


def read_raster(file_path):
    """
    读取栅格数据并返回所有波段的数据数组、投影信息、地理变换和NoData值。
    支持单波段和多波段（如RGB）数据。

    返回:
    data - 形状为 [bands, height, width] 的NumPy数组
    """
    with rasterio.open(file_path) as dataset:
        # 读取所有波段
        data = dataset.read()
        band_count = dataset.count

        # 获取每个波段的NoData值（可能不同）
        nodata_values = [dataset.nodatavals[i] for i in range(band_count)]

        projection = dataset.crs
        geotransform = dataset.transform

        print(f"读取栅格: {file_path}")
        print(f"波段数: {band_count}")
        print(f"数据形状: {data.shape}")
        print(f"NoData值: {nodata_values}")

    return data, projection, geotransform, nodata_values, band_count


def write_raster(output_path, data, projection, geotransform, nodata_values):
    """
    将处理后的数据写入新的栅格文件。
    支持单波段和多波段（如RGB）数据。

    参数:
    data - 形状为 [bands, height, width] 的NumPy数组
    """
    # 确保数据是3D数组 [bands, height, width]
    if len(data.shape) == 2:
        data = data.reshape(1, *data.shape)

    band_count, height, width = data.shape

    # 确保nodata_values是列表且长度与波段数一致
    if not isinstance(nodata_values, (list, tuple)):
        nodata_values = [nodata_values] * band_count

    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=band_count,
        dtype=data.dtype,
        crs=projection,
        transform=geotransform,
        nodata=nodata_values[0],  # rasterio限制: 所有波段使用相同的nodata值
    ) as dataset:
        for i in range(band_count):
            dataset.write(data[i], i + 1)  # 波段索引从1开始

    print(f"写入栅格: {output_path}")
    print(f"波段数: {band_count}")
    print(f"数据形状: {data.shape}")


def fill_nodata_with_mean(data, nodata_values, per_band=True, fill_value="mean"):
    """
    填充NoData值（支持多种填充策略）。

    这是一个通用的NoData填充函数，支持均值填充、固定值填充等多种策略。
    适用于单波段和多波段栅格数据（如InSAR形变、DEM、RGB图像等）。

    参数:
    ----------
    data : numpy.ndarray
        形状为 [bands, height, width] 的NumPy数组

    nodata_values : list or tuple
        NoData值列表，每个波段一个。可以是数值或None/NaN

    per_band : bool, optional (default=True)
        是否为每个波段单独计算填充值
        - True: 为每个波段分别计算均值（推荐用于多波段图像）
        - False: 使用所有波段的全局均值
        注意: 当fill_value为数值时，此参数不影响结果

    fill_value : str or numeric, optional (default="mean")
        填充策略：
        - "mean": 使用均值填充（根据per_band参数决定是波段均值还是全局均值）
        - 数值 (如 0, -9999等): 使用指定的固定值填充所有NoData

    返回:
    -------
    filled_data : numpy.ndarray
        填充后的数据，形状与输入相同

    total_filled : int
        已填充的NoData像素总数

    示例:
    -----
    >>> # 使用均值填充（按波段）
    >>> filled, count = fill_nodata_with_mean(data, nodata_vals, fill_value="mean")
    >>>
    >>> # 使用0填充
    >>> filled, count = fill_nodata_with_mean(data, nodata_vals, fill_value=0)
    >>>
    >>> # 使用全局均值填充
    >>> filled, count = fill_nodata_with_mean(data, nodata_vals, per_band=False)
    """
    # 确保数据是3D数组 [bands, height, width]
    if len(data.shape) == 2:
        data = data.reshape(1, *data.shape)
        if not isinstance(nodata_values, (list, tuple)):
            nodata_values = [nodata_values]

    # 创建数据副本
    filled_data = data.copy()
    band_count = filled_data.shape[0]

    # 确保nodata_values是列表且长度与波段数一致
    if len(nodata_values) < band_count:
        nodata_values = nodata_values * band_count

    # 对每个波段分别处理
    total_filled = 0

    # ========================================
    # 策略1: 使用固定数值填充（如0填充）
    # ========================================
    if fill_value != "mean":
        print(f"填充策略: 使用固定值 {fill_value} 填充所有NoData")

        for i in range(band_count):
            band_data = filled_data[i]
            nodata_value = nodata_values[i]

            # 创建NoData掩码
            nodata_mask = _create_nodata_mask(band_data, nodata_value)

            nodata_count = np.sum(nodata_mask)
            if nodata_count == 0:
                print(f"  波段 {i+1}: 没有NoData值，无需填充")
                continue

            # 使用固定值填充
            band_data[nodata_mask] = fill_value
            total_filled += nodata_count
            print(f"  波段 {i+1}: 已用 {fill_value} 填充 {nodata_count} 个NoData像素")

        # 如果输入是单波段，恢复原始形状
        if data.shape[0] == 1 and len(data.shape) == 3:
            filled_data = filled_data[0]

        return filled_data, total_filled

    # ========================================
    # 策略2: 使用均值填充（原有逻辑）
    # ========================================
    print(f"填充策略: 使用均值填充 (per_band={per_band})")

    if per_band:
        # 为每个波段分别计算均值
        for i in range(band_count):
            band_data = filled_data[i]
            nodata_value = nodata_values[i]

            # 创建NoData掩码
            nodata_mask = _create_nodata_mask(band_data, nodata_value)

            nodata_count = np.sum(nodata_mask)
            if nodata_count == 0:
                print(f"  波段 {i+1}: 没有NoData值，无需填充")
                continue

            # 计算该波段有效数据的均值
            valid_data = band_data[~nodata_mask]
            if valid_data.size > 0:
                band_mean = np.mean(valid_data)
                print(f"  波段 {i+1}: 均值 = {band_mean:.4f}, 已填充 {nodata_count} 个NoData像素")

                # 用波段均值填充该波段的NoData值
                band_data[nodata_mask] = band_mean
                total_filled += nodata_count
            else:
                # 如果该波段没有有效值，填充为0
                band_data[nodata_mask] = 0
                total_filled += nodata_count
                print(f"  ⚠️ 警告 - 波段 {i+1}: 没有有效值，降级使用0填充 ({nodata_count} 个像素)")
    else:
        # 计算所有波段的全局均值

        # 创建所有波段的组合掩码
        combined_mask = np.zeros_like(filled_data[0], dtype=bool)

        for i in range(band_count):
            band_mask = _create_nodata_mask(filled_data[i], nodata_values[i])
            combined_mask = combined_mask | band_mask

        # 计算所有有效像素（任何波段）的全局均值
        valid_pixels = []
        for i in range(band_count):
            band_data = filled_data[i]
            # 只考虑非掩码区域的像素
            valid_values = band_data[~combined_mask]
            if valid_values.size > 0:
                valid_pixels.extend(valid_values)

        if valid_pixels:
            global_mean = np.mean(valid_pixels)
            print(f"  全局均值 (所有波段): {global_mean:.4f}")

            # 用全局均值填充所有波段的NoData值
            for i in range(band_count):
                band_data = filled_data[i]
                band_mask = _create_nodata_mask(band_data, nodata_values[i])

                nodata_count = np.sum(band_mask)
                band_data[band_mask] = global_mean
                total_filled += nodata_count

            print(f"  已用全局均值填充所有波段共 {total_filled} 个NoData像素")
        else:
            # 如果没有有效值，所有NoData填充为0
            for i in range(band_count):
                band_data = filled_data[i]
                band_mask = _create_nodata_mask(band_data, nodata_values[i])

                nodata_count = np.sum(band_mask)
                band_data[band_mask] = 0
                total_filled += nodata_count

            print(f"  ⚠️ 警告: 所有波段没有有效值，降级使用0填充 (共 {total_filled} 个像素)")

    # 如果输入是单波段，恢复原始形状
    if data.shape[0] == 1 and len(data.shape) == 3:
        filled_data = filled_data[0]

    return filled_data, total_filled


def process_raster(input_path, output_path, fill_method="per_band", fill_value="mean"):
    """
    处理栅格数据，填充NoData值（主接口函数）。

    这是用户的主要接口函数，提供了灵活的NoData填充选项。
    支持InSAR形变数据、DEM、多光谱/RGB图像等各类栅格数据。

    参数:
    ----------
    input_path : str
        输入栅格文件路径（支持GDAL可读的所有格式，如.tif、.img等）

    output_path : str
        输出栅格文件路径（自动创建目录）

    fill_method : str, optional (default="per_band")
        填充方法（仅当fill_value="mean"时生效）：
        - "per_band": 对每个波段分别计算均值填充（推荐用于多光谱图像）
        - "global_mean": 使用所有波段的全局均值填充（适用于同质性强的数据）

    fill_value : str or numeric, optional (default="mean")
        填充策略：
        - "mean": 使用均值填充（根据fill_method参数决定具体策略）
        - 数值 (如 0, -9999, np.nan等): 使用指定的固定值填充

    返回:
    -------
    filled_data : numpy.ndarray
        填充后的数据数组

    示例:
    -----
    >>> # 示例1: 使用波段均值填充InSAR数据
    >>> process_raster("insar.tif", "insar_filled.tif")
    >>>
    >>> # 示例2: 使用0填充DEM数据
    >>> process_raster("dem.tif", "dem_filled.tif", fill_value=0)
    >>>
    >>> # 示例3: 使用全局均值填充RGB图像
    >>> process_raster("rgb.tif", "rgb_filled.tif", fill_method="global_mean")
    >>>
    >>> # 示例4: 使用-9999填充（常用于某些GIS软件）
    >>> process_raster("data.tif", "data_filled.tif", fill_value=-9999)
    """
    # 读取数据
    data, projection, geotransform, nodata_values, band_count = read_raster(input_path)

    print("=" * 70)
    print(f"开始处理: {input_path}")
    print(f"填充方法: {fill_method}")
    print(f"填充值: {fill_value}")
    print("=" * 70)

    # 计算原始NoData像素数量
    total_nodata = 0
    for i in range(band_count):
        nodata_value = nodata_values[i]
        if nodata_value is None:
            nodata_values[i] = np.nan  # 如果NoData值未定义，则使用NaN

        if np.isnan(nodata_values[i]):
            band_nodata = np.sum(np.isnan(data[i]))
        else:
            band_nodata = np.sum(data[i] == nodata_values[i])

        print(f"波段 {i+1} 原始NoData像素数: {band_nodata}")
        total_nodata += band_nodata

    print(f"所有波段共有 {total_nodata} 个NoData像素")

    # 根据方法填充NoData值
    if fill_method == "per_band":
        filled_data, pixels_filled = fill_nodata_with_mean(data, nodata_values, per_band=True, fill_value=fill_value)
    elif fill_method == "global_mean":
        filled_data, pixels_filled = fill_nodata_with_mean(data, nodata_values, per_band=False, fill_value=fill_value)
    else:
        print(f"⚠️ 未知的填充方法 '{fill_method}'，使用默认的'per_band'方法")
        filled_data, pixels_filled = fill_nodata_with_mean(data, nodata_values, per_band=True, fill_value=fill_value)

    # 输出处理结果
    print("=" * 70)
    print(f"✓ 处理完成！共填充 {pixels_filled} 个NoData像素")
    print(f"✓ 输出到: {output_path}")
    print("=" * 70)

    # 写入结果
    write_raster(output_path, filled_data, projection, geotransform, nodata_values)

    return filled_data


# ============================================================================
# 便捷函数 (Convenience Functions)
# ============================================================================


def process_raster_with_zero(input_path, output_path):
    """
    使用0填充NoData值的便捷函数。

    这是最常用的便捷函数，适合需要将NoData区域归零的场景。
    例如：DEM处理、二值化图像、mask生成等。

    参数:
    ----------
    input_path : str
        输入栅格文件路径
    output_path : str
        输出栅格文件路径

    返回:
    -------
    filled_data : numpy.ndarray
        填充后的数据数组

    示例:
    -----
    >>> # 快速将DEM的NoData区域填充为0
    >>> process_raster_with_zero("dem.tif", "dem_zero_filled.tif")
    """
    return process_raster(input_path, output_path, fill_value=0)


# ============================================================================
# 批量处理函数 (Batch Processing)
# ============================================================================


def batch_process(
    input_dir, output_dir, pattern="*.tif", fill_method="per_band", fill_value="mean", output_suffix="_filled"
):
    """
    批量处理目录中的栅格文件。

    自动遍历指定目录，对所有匹配的文件应用相同的填充策略。
    适合大规模数据处理场景。

    参数:
    ----------
    input_dir : str
        输入文件目录
    output_dir : str
        输出文件目录（自动创建）
    pattern : str, optional (default="*.tif")
        文件匹配模式（支持通配符）
    fill_method : str, optional (default="per_band")
        填充方法（"per_band" 或 "global_mean"）
    fill_value : str or numeric, optional (default="mean")
        填充值（"mean" 或数值）
    output_suffix : str, optional (default="_filled")
        输出文件名后缀

    示例:
    -----
    >>> # 批量使用0填充
    >>> batch_process("input_dir/", "output_dir/", fill_value=0)
    >>>
    >>> # 批量使用均值填充
    >>> batch_process("input_dir/", "output_dir/", fill_value="mean")
    """
    print("\n" + "=" * 70)
    print("开始批量处理")
    print("=" * 70)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"匹配模式: {pattern}")
    print(f"填充方法: {fill_method}")
    print(f"填充值: {fill_value}")
    print("=" * 70 + "\n")

    # 获取匹配的文件
    input_files = glob.glob(os.path.join(input_dir, pattern))

    if not input_files:
        print(f"⚠️ 警告: 在 {input_dir} 中未找到匹配 {pattern} 的文件")
        return

    print(f"找到 {len(input_files)} 个文件待处理\n")

    success_count = 0
    fail_count = 0

    for idx, input_file in enumerate(input_files, 1):
        # 构建输出文件路径
        filename = os.path.basename(input_file)
        name_without_ext = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]
        output_filename = f"{name_without_ext}{output_suffix}{ext}"
        output_file = os.path.join(output_dir, output_filename)

        # 处理文件
        print(f"\n[{idx}/{len(input_files)}] 处理: {filename}")
        try:
            process_raster(input_file, output_file, fill_method, fill_value)
            success_count += 1
            print("✓ 成功")
        except Exception as e:
            fail_count += 1
            print(f"✗ 失败: {str(e)}")

    # 汇总结果
    print("\n" + "=" * 70)
    print("批量处理完成")
    print("=" * 70)
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    """
    ========================================================================
    批量处理示例
    ========================================================================
    """

    # ========================================================================
    # 推荐使用：批量处理Yajiang数据集（使用0填充NoData）
    # ========================================================================
    # input_directory = r"E:\Document\paper_library\5th_paper_InSAR\datasets\01_raw\Yajiang_tif\meta"
    # output_directory = r"E:\Document\paper_library\5th_paper_InSAR\datasets\01_raw\Yajiang_tif\meta_nodata"

    # # 批量使用0填充（添加_nodata后缀）
    # batch_process(input_directory, output_directory, pattern="*.tif", fill_value=0, output_suffix="_nodata")

    # 说明：
    # - 只处理Yajiang_tif目录下的.tif文件（不包含子目录）
    # - 使用0填充所有NoData区域
    # - 输出到Yajiang_tif_nodata目录，文件名添加_nodata后缀
    #   例如: image.tif -> image_nodata.tif
    # - 如果不想添加后缀，可以设置 output_suffix=""

    # ========================================================================
    # 其他批量处理示例（可选）
    # ========================================================================

    # 示例1：使用均值填充而不是0填充
    # batch_process(input_directory, output_directory,
    #               pattern="*.tif", fill_value="mean", output_suffix="_mean")

    # 示例2：使用自定义值填充（如0）
    # batch_process(input_directory, output_directory,
    #               pattern="*.tif", fill_value=0, output_suffix="_nodata")

    # 示例3：不添加后缀（保持原文件名）
    # batch_process(input_directory, output_directory,
    #               pattern="*.tif", fill_value=0, output_suffix="")

    # 示例4：只处理特定命名模式的文件
    # batch_process(input_directory, output_directory,
    #               pattern="InSAR_*.tif", fill_value=0, output_suffix="_nodata")

    # 示例5：处理其他目录的数据（已注释）
    # other_input = r"E:\Document\paper_library\5th_paper_InSAR\datasets\01_raw\Yajiang_tif_10m"
    # other_output = r"E:\Document\paper_library\5th_paper_InSAR\datasets\01_raw\02_aligned_grid"
    # batch_process(other_input, other_output, pattern="*.tif", fill_value=0, output_suffix="_mean")

    # ========================================================================
    # 单文件处理示例
    # ========================================================================
    input_file = (
        r"E:\Document\paper_library\5th_paper_InSAR\datasets\01_raw\Yajiang_tif_10m\Yajiang_aspect_clip_10m.tif"
    )
    output_file = (
        r"E:\Document\paper_library\5th_paper_InSAR\datasets\02_aligned_grid\Yajiang_aspect_clip_10m_nodata.tif"
    )

    # 使用均值填充（按波段独立计算均值）
    # process_raster(input_file, output_file, fill_value="mean")

    # 或者使用0填充（取消注释下一行）
    process_raster(input_file, output_file, fill_value=0)

    print("\n✓ 单文件处理完成！")
