import arcpy
import os

def batch_resample_align():
    # ================= 配置区域 =================
    # 1. 输入数据所在的文件夹
    input_folder = r"E:\Document\paper_library\5th_paper_InSAR\datasets\Yajiang_tif"
    
    # 2. 参考影像路径 (作为捕捉栅格、范围和分辨率的基准)
    ref_raster_path = r"E:\Document\paper_library\5th_paper_InSAR\datasets\Yajiang_tif\S2_NDVI_20250515_20250930.tif"
    
    # 3. 输出文件夹路径
    output_folder = r"E:\Document\paper_library\5th_paper_InSAR\datasets\Yajiang_tif\Yajiang_tif_10m"
    
    # 4. 重采样方法: "NEAREST" (分类数据), "BILINEAR" (连续数据), "CUBIC" (平滑连续数据)
    # 考虑到你是InSAR/NDVI类数据，推荐 BILINEAR 或 CUBIC
    resample_alg = "BILINEAR" 
    # ===========================================

    # 检查并创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")

    # 获取参考影像的属性
    print("正在读取参考影像属性...")
    ref_desc = arcpy.Describe(ref_raster_path)
    
    # 获取像元大小 (假设是方形像元，取宽度)
    cell_size_x = ref_desc.meanCellWidth
    cell_size_y = ref_desc.meanCellHeight
    # 转换为字符串形式 "10 10" 供工具使用
    target_cell_size = f"{cell_size_x} {cell_size_y}"
    
    print(f"目标分辨率: {target_cell_size}")
    print(f"目标坐标系: {ref_desc.spatialReference.name}")

    # ================= 关键步骤：设置全局环境变量 =================
    # 1. 输出坐标系与参考一致
    arcpy.env.outputCoordinateSystem = ref_desc.spatialReference
    # 2. 捕捉栅格 (Snap Raster): 保证像元网格对齐
    arcpy.env.snapRaster = ref_raster_path
    # 3. 处理范围 (Extent): 强制输出范围与参考影像一致（统一行列数的关键）
    arcpy.env.extent = ref_raster_path
    # 4. 允许覆盖同名文件
    arcpy.env.overwriteOutput = True

    # 获取文件夹内所有的 tif 文件
    arcpy.env.workspace = input_folder
    raster_list = arcpy.ListRasters("*", "TIF")

    print(f"找到 {len(raster_list)} 个待处理文件，开始处理...\n")

    for raster in raster_list:
        # 构造完整路径
        in_raster_path = os.path.join(input_folder, raster)
        
        # 如果是参考影像本身，决定是否跳过或复制。这里选择跳过，因为它已经是标准了
        if os.path.normpath(in_raster_path) == os.path.normpath(ref_raster_path):
            print(f"跳过参考影像本身: {raster}")
            continue

        # 构造输出文件名：原名 + _10m.tif
        file_name_no_ext = os.path.splitext(raster)[0]
        out_raster_name = f"{file_name_no_ext}_10m.tif"
        out_raster_path = os.path.join(output_folder, out_raster_name)

        try:
            print(f"正在重采样: {raster} -> {out_raster_name}")
            
            # 执行重采样
            # 注意：因为设置了环境变量 extent 和 snapRaster，
            # 即使这里只指定了 cell size，输出结果的行列数也会被环境变量强制对齐到参考影像。
            arcpy.management.Resample(
                in_raster=in_raster_path,
                out_raster=out_raster_path,
                cell_size=target_cell_size,
                resampling_type=resample_alg
            )
            print("  -> 完成")
            
        except Exception as e:
            print(f"  -> 处理 {raster} 时出错: {e}")

    print("\n所有处理已完成。")

# 执行函数
if __name__ == '__main__':
    batch_resample_align()