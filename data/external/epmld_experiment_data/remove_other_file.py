import os

# 获取当前目录
current_directory = os.getcwd()

# 递归遍历目录
for root, dirs, files in os.walk(current_directory):
    for filename in files:
        file_path = os.path.join(root, filename)

        # 获取文件扩展名
        _, ext = os.path.splitext(filename)

        # 如果文件扩展名不是 .stim 或 .dat，则删除该文件
        if ext.lower() not in ['.stim', '.dat', '.py', '.ipynb', '.md']:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
