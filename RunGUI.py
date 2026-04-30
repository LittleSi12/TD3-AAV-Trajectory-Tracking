"""
启动搜救无人机轨迹跟随可视化 GUI。
用法：python RunGUI.py

Author: Little Si
Date:   2026-04
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from Visualization_GUI.app import main

if __name__ == "__main__":
    main()
