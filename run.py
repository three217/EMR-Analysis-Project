#run
import os
import sys
import subprocess

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    main_script_path = os.path.join(current_dir, "_main_2.0.py")
    
    if not os.path.exists(main_script_path):
        print(f"❌ 错误：找不到 _main_2.0.py！")
        print(f"   预期路径：{main_script_path}")
        print(f"   请确保 run.py 和 _main_2.0.py 在同一个文件夹下！")
        input("按回车键退出...")
        sys.exit(1)
    
    print(f"🚀 正在启动住院病历结构化智能体...")
    print(f"📂 主程序路径：{main_script_path}")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", main_script_path,
            "--server.port", "8501"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 程序已停止")
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败：{e}")
        input("按回车键退出...")

if __name__ == "__main__":
    main()
