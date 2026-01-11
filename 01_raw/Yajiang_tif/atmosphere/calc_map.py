import arcpy
import os
import shutil
from arcpy.sa import *

# ================= 核心配置区 =================
# 1. NC文件所在文件夹
input_folder = r"E:\Document\paper_library\5th_paper_InSAR\datasets\Yajiang_tif\atmosphere\test"

# 2. 参考栅格路径 (您的研究区)
ref_raster_path = r"E:\Document\paper_library\5th_paper_InSAR\datasets\Yajiang_tif\atmosphere\prec_Layer_clip.tif"

# 3. 最终输出结果
output_raster = r"E:\Document\paper_library\5th_paper_InSAR\datasets\Yajiang_tif\atmosphere\Mean_Precip_1980_2024.tif"

# 4. 临时工作区 (请确保磁盘空间充足，因为会生成中间TIF)
temp_workspace = r"E:\Document\paper_library\5th_paper_InSAR\datasets\Yajiang_tif\atmosphere\temp_process"

# 5. NC变量名 (非常重要！请确认是 'prec', 'precipitation' 还是其他)
# 根据您的文件名 'ChinaMet_001deg_prec_1985.nc'，猜测变量名为 'prec'
variable_name = "prec"
# ============================================


def safe_process():
    # --- 环境初始化 ---
    print("正在初始化环境...")
    arcpy.CheckOutExtension("Spatial")
    arcpy.env.overwriteOutput = True

    # 重建临时文件夹
    if os.path.exists(temp_workspace):
        shutil.rmtree(temp_workspace)
    os.makedirs(temp_workspace)

    # 获取参考栅格信息
    desc_ref = arcpy.Describe(ref_raster_path)
    ref_sr = desc_ref.spatialReference
    print(f"参考坐标系: {ref_sr.name}")

    nc_files = [f for f in os.listdir(input_folder) if f.endswith(".nc")]
    valid_clips = []

    print(f"--- 启动 '先转TIF后裁剪' 策略 (待处理: {len(nc_files)} 个) ---")

    for index, nc_file in enumerate(nc_files):
        nc_path = os.path.join(input_folder, nc_file)
        print(f"\n[{index+1}/{len(nc_files)}] 处理: {nc_file}")

        # 临时文件路径
        temp_raw_tif = os.path.join(temp_workspace, f"raw_{index}.tif")
        temp_final_clip = os.path.join(temp_workspace, f"clip_{index}.tif")

        try:
            # ========================================================
            # 第一阶段：格式转换 (NC -> 物理 TIF)
            # 目标：彻底脱离 NetCDF 环境，生成一个实实在在的栅格文件
            # ========================================================

            # 1. 读取维度
            nc_fp = arcpy.NetCDFFileProperties(nc_path)
            dims = nc_fp.getDimensions()
            # 自动探测经纬度维度名
            x_dim = next((d for d in dims if "lon" in d.lower() or "x" in d.lower()), "lon")
            y_dim = next((d for d in dims if "lat" in d.lower() or "y" in d.lower()), "lat")

            # 2. 创建内存图层
            memory_layer = "temp_nc_layer"
            arcpy.md.MakeNetCDFRasterLayer(nc_path, variable_name, x_dim, y_dim, memory_layer)

            # 3. 【核心修复】立即保存为 TIF，不设置任何裁剪环境！
            # 我们先不设 extent，让它把全球（或全幅）数据完整写出。
            # 这样避免了因坐标微小偏差导致的“裁剪为空”。
            print("  -> 正在转换为中间 TIF (物理落地)...")
            arcpy.management.CopyRaster(memory_layer, temp_raw_tif)

            # 4. 强制定义中间 TIF 的坐标系 (NC通常是 WGS84)
            # 如果您的 NC 确定是 WGS84，这里显式定义一次以防万一
            arcpy.management.DefineProjection(temp_raw_tif, arcpy.SpatialReference(4326))

            # ========================================================
            # 第二阶段：空间裁剪 (TIF -> Clip TIF)
            # 目标：现在它是 TIF 了，我们可以像处理普通数据一样处理它
            # ========================================================

            if os.path.exists(temp_raw_tif):
                print("  -> 正在执行精确裁剪...")

                # 设置严格的对齐环境
                with arcpy.EnvManager(
                    snapRaster=ref_raster_path,
                    extent=desc_ref.extent,
                    cellSize=ref_raster_path,
                    outputCoordinateSystem=ref_sr,
                ):

                    # 使用 ExtractByMask (比 Clip 更适合不规则边界，且处理 NoData 更好)
                    out_extract = ExtractByMask(temp_raw_tif, ref_raster_path)
                    out_extract.save(temp_final_clip)

                # ========================================================
                # 第三阶段：验证
                # ========================================================
                if os.path.exists(temp_final_clip):
                    # 重新计算统计值
                    arcpy.management.CalculateStatistics(temp_final_clip)
                    max_res = arcpy.GetRasterProperties_management(temp_final_clip, "MAXIMUM")
                    max_val = float(max_res.getOutput(0))

                    # 检查数据是否有效 (根据降雨量逻辑，应该 >= 0)
                    if max_val > -9999:  # 只要不是极小的 NoData 标记值
                        valid_clips.append(temp_final_clip)
                        print(f"  -> 成功: Max = {max_val:.2f}")

                        # 成功后，删除那个巨大的中间 raw_tif 以节省空间
                        try:
                            os.remove(temp_raw_tif)
                        except:
                            pass
                    else:
                        print(f"  -> 警告: 裁剪结果全是 NoData (Max: {max_val})")
                else:
                    print("  -> 错误: 裁剪文件未能生成")
            else:
                print("  -> 错误: 中间 TIF 转换失败")

        except Exception as e:
            print(f"  -> 处理单文件出错: {e}")
            # 如果出错，保留中间文件以便排查，不删除

    # ========================================================
    # 第四阶段：聚合计算
    # ========================================================
    if not valid_clips:
        raise Exception("没有生成任何有效文件，请检查 NC 变量名或原始数据。")

    print(f"\n正在计算 {len(valid_clips)} 年的平均降水量...")
    out_mean = CellStatistics(valid_clips, "MEAN", "DATA")
    out_mean.save(output_raster)

    print(f"\n【处理完成】")
    print(f"输出路径: {output_raster}")

    # 最终清理临时目录 (可选)
    # shutil.rmtree(temp_workspace)


if __name__ == "__main__":
    try:
        safe_process()
    except Exception as e:
        import traceback

        print(traceback.format_exc())
