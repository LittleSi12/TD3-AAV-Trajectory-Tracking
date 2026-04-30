"""
CSV 读写工具。

Author: Little Si
Date:   2026-04
"""

import pandas as pd
import numpy as np

class DataIO:
    @staticmethod
    def save_trajectory(trajectory, filename="trajectory_points.csv"):
        """保存轨迹到CSV文件"""
        df = pd.DataFrame(trajectory, columns=['x', 'y'])
        df.to_csv(filename, index=False)
        print(f"轨迹已保存到：{filename}")
    
    @staticmethod
    def load_trajectory(filename="trajectory_points.csv"):
        """从CSV文件加载轨迹"""
        try:
            df = pd.read_csv(filename)
            trajectory = df[['x', 'y']].values
            print(f"轨迹已从 {filename} 加载，共 {len(trajectory)} 个点")
            return trajectory
        except FileNotFoundError:
            print(f"文件 {filename} 不存在")
            return None
    
    @staticmethod
    def save_results(results, filename="results.csv"):
        """保存评估结果到CSV文件"""
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"结果已保存到：{filename}")
    
    @staticmethod
    def load_results(filename="results.csv"):
        """从CSV文件加载评估结果"""
        try:
            df = pd.read_csv(filename)
            print(f"结果已从 {filename} 加载")
            return df
        except FileNotFoundError:
            print(f"文件 {filename} 不存在")
            return None
