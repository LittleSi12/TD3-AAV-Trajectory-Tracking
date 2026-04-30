"""
启动搜救无人机轨迹可视化 GUI。
用法：python RunGUI.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from Visualization_GUI.app import main

if __name__ == "__main__":
    main()
