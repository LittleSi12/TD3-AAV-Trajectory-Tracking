"""
搜救无人机轨迹可视化 GUI — 三阶段交互式仿真
"""

import sys, os, time, threading
import numpy as np
import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["SimSun", "Times New Roman"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["mathtext.fontset"] = "stix"
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from UAV_Environment.TrajectoryGenerator import TrajectoryGenerator

C_BG="#f0f2f5"; C_CARD="#ffffff"; C_CARD2="#f7f8fa"; C_BORDER="#e0e3ea"
C_TEXT="#1a1a2e"; C_DIM="#8c8fa3"; C_ACCENT="#4361ee"; C_GREEN="#10b981"
C_ORANGE="#f59e0b"; C_PURPLE="#8b5cf6"; C_RED="#ef4444"; C_LOG="#1e40af"

DISASTER_TYPES = [
    ("火灾","🔥","#ef4444"),("建筑倒塌","⚠","#f59e0b"),
    ("洪水","🌊","#3b82f6"),("滑坡","⛰","#8b5cf6"),("化学泄漏","☢","#10b981"),
]

class MetricCard(ctk.CTkFrame):
    def __init__(self, master, label, value, color, **kw):
        super().__init__(master, fg_color=C_CARD, corner_radius=10,
                         border_width=1, border_color=C_BORDER, **kw)
        self.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self, text=label, font=("SimSun",17),
                     text_color=C_DIM, anchor="w"
                     ).grid(row=0,column=0,sticky="w",padx=12,pady=(8,0))
        self.val_label = ctk.CTkLabel(self, text=value,
                     font=("Consolas",30,"bold"), text_color=color, anchor="w")
        self.val_label.grid(row=1,column=0,sticky="w",padx=12,pady=(0,4))
        ctk.CTkFrame(self, fg_color=color, height=3, corner_radius=0
                     ).grid(row=2,column=0,sticky="ew")
    def set_value(self, v): self.val_label.configure(text=v)

class TrajectoryVisualizerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        self.title("搜救无人机轨迹跟随可视化系统")
        self.geometry("1280x780")
        self.minsize(1100,700)
        self.configure(fg_color=C_BG)
        # 启动后自动最大化
        self.after(100, lambda: self.state("zoomed"))
        self.generator=None; self.sim_result=None; self._anim_id=None
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()

    def _on_close(self):
        """关闭窗口时彻底退出进程。"""
        if self._anim_id is not None:
            self.after_cancel(self._anim_id)
            self._anim_id = None
        plt.close("all")
        self.destroy()
        import sys
        sys.exit(0)

    def _build_ui(self):
        self.grid_columnconfigure(0,weight=5)
        self.grid_columnconfigure(1,weight=3)
        self.grid_rowconfigure(1,weight=1)

        # Row 0: 按钮(左) + 指标卡(右), 同一行, 按钮与卡片等高
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.grid(row=0,column=0,columnspan=2,sticky="ew",padx=16,pady=(8,4))

        self.btn_env = ctk.CTkButton(top, text="① 环境设置",
            font=("SimSun",18,"bold"), fg_color=C_ACCENT, hover_color="#3b56cc",
            text_color="white", corner_radius=8, height=66, width=180,
            command=self._on_env_setup)
        self.btn_env.pack(side="left",padx=(0,6))
        self.btn_plan = ctk.CTkButton(top, text="② 轨迹规划",
            font=("SimSun",18,"bold"), fg_color="#94a3b8", text_color="white",
            corner_radius=8, height=66, width=180, state="disabled",
            command=self._on_plan)
        self.btn_plan.pack(side="left",padx=(0,6))
        self.btn_start = ctk.CTkButton(top, text="③ 任务开始",
            font=("SimSun",18,"bold"), fg_color="#94a3b8", text_color="white",
            corner_radius=8, height=66, width=180, state="disabled",
            command=self._on_start)
        self.btn_start.pack(side="left",padx=(0,16))

        self.card_in5m = MetricCard(top,"5m内比例","--",C_PURPLE)
        self.card_in5m.pack(side="right",padx=3)
        self.card_avg = MetricCard(top,"平均距离","--",C_ORANGE)
        self.card_avg.pack(side="right",padx=3)
        self.card_dist = MetricCard(top,"当前距离","--",C_ACCENT)
        self.card_dist.pack(side="right",padx=3)
        self.card_progress = MetricCard(top,"任务进度","--",C_GREEN)
        self.card_progress.pack(side="right",padx=3)

        # Row 1 left: 2D/3D 标签页
        self.tabview = ctk.CTkTabview(self, fg_color=C_CARD, corner_radius=12,
            border_width=1, border_color=C_BORDER,
            segmented_button_fg_color=C_CARD2,
            segmented_button_selected_color=C_ACCENT,
            segmented_button_unselected_color=C_CARD2,
            segmented_button_selected_hover_color="#3b56cc",
            segmented_button_unselected_hover_color=C_BORDER)
        self.tabview.grid(row=1,column=0,sticky="nsew",padx=(16,6),pady=(4,12))
        self.tabview._segmented_button.configure(font=("SimSun",18,"bold"))
        self.tab_2d = self.tabview.add("    2D 视角    ")
        self.tab_3d = self.tabview.add("    3D 视角    ")
        ctk.CTkLabel(self.tab_2d, text="点击「环境设置」开始",
                     font=("SimSun",18), text_color=C_DIM
                     ).place(relx=0.5,rely=0.5,anchor="center")
        ctk.CTkLabel(self.tab_3d, text="点击「环境设置」开始",
                     font=("SimSun",18), text_color=C_DIM
                     ).place(relx=0.5,rely=0.5,anchor="center")

        # Row 1 right: 日志
        lp = ctk.CTkFrame(self, fg_color=C_CARD, corner_radius=12,
                           border_width=1, border_color=C_BORDER)
        lp.grid(row=1,column=1,sticky="nsew",padx=(6,16),pady=(4,12))
        lp.grid_columnconfigure(0,weight=1); lp.grid_rowconfigure(1,weight=1)
        log_header = ctk.CTkFrame(lp, fg_color="transparent")
        log_header.grid(row=0,column=0,sticky="ew",padx=14,pady=(10,4))
        ctk.CTkLabel(log_header, text="运行日志", font=("SimSun",16,"bold"),
                     text_color=C_TEXT).pack(side="left")
        ctk.CTkButton(log_header, text="清空", font=("SimSun",12),
                      fg_color="#94a3b8", hover_color="#64748b", text_color="white",
                      corner_radius=6, height=28, width=60,
                      command=self._clear_log).pack(side="right")
        self.log_box = ctk.CTkTextbox(lp, fg_color=C_CARD2, text_color="#64748b",
            font=("Consolas",20), corner_radius=8, border_width=1,
            border_color=C_BORDER, wrap="none")
        self.log_box.grid(row=1,column=0,sticky="nsew",padx=10,pady=(0,10))
        # 配置富文本 tag
        tb = self.log_box._textbox
        tb.tag_configure("title",    foreground="#1e293b", font=("SimSun", 21, "bold"))
        tb.tag_configure("sep",      foreground="#cbd5e1")
        tb.tag_configure("key",      foreground="#475569", font=("SimSun", 20))
        tb.tag_configure("val",      foreground="#1e40af", font=("Consolas", 20, "bold"))
        tb.tag_configure("success",  foreground="#059669", font=("SimSun", 20, "bold"))
        tb.tag_configure("error",    foreground="#dc2626", font=("SimSun", 20, "bold"))
        tb.tag_configure("info",     foreground="#64748b", font=("SimSun", 20))
        tb.tag_configure("data",     foreground="#334155", font=("Consolas", 19))
        tb.tag_configure("zone",     foreground="#475569", font=("Consolas", 19))
        tb.tag_configure("dim",      foreground="#94a3b8", font=("Consolas", 17))
        tb.tag_configure("init",     foreground="#64748b", font=("SimSun", 24))
        self._log_styled("系统就绪，等待操作...", "init")
        self._log("")

        self.fig_2d=None; self.ax_2d=None; self.canvas_2d=None
        self.fig_3d=None; self.ax_3d=None; self.canvas_3d=None

    def _log(self, text):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text+"\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _log_styled(self, text, tag=None):
        """插入带 tag 样式的一行文本。"""
        self.log_box.configure(state="normal")
        if tag:
            tb = self.log_box._textbox
            start = tb.index("end-1c")
            tb.insert("end", text + "\n")
            end = tb.index("end-1c")
            tb.tag_add(tag, start, end)
        else:
            self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _log_kv(self, key, value):
        """插入一行键值对，key 和 value 分别着色。"""
        self.log_box.configure(state="normal")
        tb = self.log_box._textbox
        # key 部分
        s1 = tb.index("end-1c")
        tb.insert("end", f"  {key}  ")
        e1 = tb.index("end-1c")
        tb.tag_add("key", s1, e1)
        # value 部分
        s2 = tb.index("end-1c")
        tb.insert("end", f"{value}\n")
        e2 = tb.index("end-1c")
        tb.tag_add("val", s2, e2)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _log_title(self, title):
        """阶段标题，带颜色装饰线。"""
        self._log("")
        self._log_styled("  ━━━━━━━━━━━━━━━━", "sep")
        self._log_styled(f"  ▶  {title}", "title")
        self._log_styled("  ━━━━━━━━━━━━━━━━", "sep")

    def _log_row(self, key, value):
        """键值行，带颜色区分。"""
        self._log_kv(key, value)

    def _log_sep(self):
        self._log("")

    def _clear_log(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0","end")
        self.log_box.configure(state="disabled")

    def _init_charts(self):
        for w in self.tab_2d.winfo_children(): w.destroy()
        for w in self.tab_3d.winfo_children(): w.destroy()
        if self.fig_2d: plt.close(self.fig_2d)
        if self.fig_3d: plt.close(self.fig_3d)

        self.fig_2d, self.ax_2d = plt.subplots(figsize=(8,8), dpi=120)
        self.fig_2d.patch.set_facecolor("#ffffff")
        ax = self.ax_2d
        ax.set_xlim(-100,1100); ax.set_ylim(-100,1100); ax.set_aspect("equal")
        ax.set_xlabel("X坐标 (m)",fontsize=24); ax.set_ylabel("Y坐标 (m)",fontsize=24)
        ax.tick_params(labelsize=20); ax.grid(True,alpha=0.3)
        self.canvas_2d = FigureCanvasTkAgg(self.fig_2d, master=self.tab_2d)
        self.canvas_2d.get_tk_widget().pack(fill="both",expand=True)

        self.fig_3d = plt.figure(figsize=(8,8), dpi=120)
        self.fig_3d.patch.set_facecolor("#ffffff")
        ax3 = self.fig_3d.add_subplot(111, projection="3d")
        self.ax_3d = ax3
        ax3.set_xlim(-100,1100); ax3.set_ylim(-100,1100)
        ax3.set_zlim(-10,100); ax3.set_zticks([])
        ax3.set_xlabel("X坐标 (m)",fontsize=22,labelpad=12)
        ax3.set_ylabel("Y坐标 (m)",fontsize=22,labelpad=12)
        ax3.set_zlabel(""); ax3.tick_params(labelsize=18)
        ax3.view_init(elev=35,azim=-50); ax3.grid(True,alpha=0.3)
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=self.tab_3d)
        self.canvas_3d.get_tk_widget().pack(fill="both",expand=True)

    def _redraw(self):
        if self.canvas_2d: self.canvas_2d.draw_idle()
        if self.canvas_3d: self.canvas_3d.draw_idle()

    # === 阶段一 ===
    def _on_env_setup(self):
        self.btn_env.configure(state="disabled")
        self.btn_plan.configure(state="disabled",fg_color="#94a3b8")
        self.btn_start.configure(state="disabled",fg_color="#94a3b8")
        def _worker():
            gen = TrajectoryGenerator()
            gen.seed = int(time.time()*1000)%(2**31)
            gen.generate()
            self.generator = gen
            self.after(0, self._draw_environment)
        threading.Thread(target=_worker, daemon=True).start()

    def _draw_environment(self):
        gen = self.generator
        self._init_charts()
        self._log_title("环境设置")
        rng = np.random.RandomState(gen.seed)
        for idx,(x,y,radius,dl,nc) in enumerate(gen.parent_points):
            name,emoji,color = DISASTER_TYPES[idx%len(DISASTER_TYPES)]
            angles = 2*np.pi*rng.rand(nc)
            dists = radius*np.sqrt(rng.rand(nc))
            cx = x+dists*np.cos(angles); cy = y+dists*np.sin(angles)
            # 2D
            self.ax_2d.scatter(cx,cy,c=color,alpha=0.4,s=12,edgecolors="none")
            self.ax_2d.add_patch(plt.Circle((x,y),radius,color=color,
                fill=True,alpha=0.08,linewidth=2,linestyle="--"))
            self.ax_2d.add_patch(plt.Circle((x,y),radius,color=color,
                fill=False,linewidth=2,linestyle="--"))
            self.ax_2d.scatter(x,y,c=color,marker="x",s=100,linewidths=2,zorder=5)
            self.ax_2d.annotate(f"{emoji} {name}\nDL:{dl}",(x,y),
                textcoords="offset points",xytext=(12,12),
                fontsize=19,color=color,fontweight="bold")
            # 3D
            self.ax_3d.scatter(cx,cy,0,c=color,alpha=0.4,s=10)
            theta = np.linspace(0,2*np.pi,80)
            self.ax_3d.plot(x+radius*np.cos(theta),y+radius*np.sin(theta),
                0,color=color,linestyle="--",linewidth=2,alpha=0.5)
            self.ax_3d.scatter(x,y,0,c=color,marker="x",s=80,zorder=5)
            # 名称补齐：用普通空格将短名称右填充到与四字名称等宽
            pad_spaces = (4 - len(name)) * 2
            padded = name + ' ' * pad_spaces
            self._log_styled(f"  [{idx+1}] {emoji} {padded}  ({x:>3d},{y:>3d})  R={radius:<3d}  DL={dl:<2d}  N={nc}", "zone")
        self._log("")
        self._log_styled("  ✅ 环境设置完成，请点击「轨迹规划」", "success")
        self.ax_3d.set_title("搜救环境 — 3D 视角",fontsize=22)
        self._redraw()
        self.btn_env.configure(state="normal")
        self.btn_plan.configure(state="normal",fg_color=C_ACCENT)

    # === 阶段二 ===
    def _on_plan(self):
        self.btn_plan.configure(state="disabled")
        gen = self.generator; sp = gen.smooth_path
        # 画轨迹线、起终点、箭头，一起闪烁
        self._traj_line, = self.ax_2d.plot(sp[:,0],sp[:,1],"g--",linewidth=2,
            label="地面搜救路径",alpha=1.0)
        self._traj_line_3d, = self.ax_3d.plot(sp[:,0],sp[:,1],0,"g--",linewidth=2,
            label="地面搜救路径",alpha=1.0)
        # 起终点
        self._start_2d = self.ax_2d.scatter(*gen.path[0],c="green",marker="o",s=150,zorder=6,label="起始点")
        self._end_2d = self.ax_2d.scatter(*gen.path[-1],c="red",marker="o",s=150,zorder=6,label="终点")
        self._start_3d = self.ax_3d.scatter(gen.path[0][0],gen.path[0][1],0,c="green",marker="o",s=150,zorder=6,label="起始点")
        self._end_3d = self.ax_3d.scatter(gen.path[-1][0],gen.path[-1][1],0,c="red",marker="o",s=150,zorder=6,label="终点")
        # 箭头
        self._arrows = []
        for i in range(1,len(sp),50):
            a = self.ax_2d.arrow(sp[i-1,0],sp[i-1,1],sp[i,0]-sp[i-1,0],sp[i,1]-sp[i-1,1],
                head_width=20,head_length=25,fc="green",ec="green",alpha=0.7)
            self._arrows.append(a)
        self.ax_2d.set_title("轨迹规划中...",fontsize=22)
        self.ax_3d.set_title("轨迹规划中... — 3D",fontsize=22)
        diffs = np.diff(sp,axis=0)
        total_len = np.sum(np.hypot(diffs[:,0],diffs[:,1]))
        self._log_title("轨迹规划")
        self._log_row("轨迹点数", str(len(sp)))
        self._log_row("路径长度", f"{total_len:.1f} m")
        self._log_row("关键点数", str(len(gen.path)))
        self._log_sep()
        self._blink_count=0; self._blink_total=6; self._do_blink()

    def _do_blink(self):
        if self._blink_count >= self._blink_total:
            # 闪烁结束后隐藏所有轨迹元素
            self._traj_line.set_alpha(0.0)
            self._traj_line_3d.set_alpha(0.0)
            self._start_2d.set_alpha(0.0); self._end_2d.set_alpha(0.0)
            self._start_3d.set_alpha(0.0); self._end_3d.set_alpha(0.0)
            for a in self._arrows: a.set_alpha(0.0)
            self.ax_2d.set_title("搜救环境 — 轨迹规划完成",fontsize=22)
            self.ax_3d.set_title("轨迹规划完成 — 3D",fontsize=22)
            self._redraw()
            self.btn_plan.configure(state="normal")
            self.btn_start.configure(state="normal",fg_color=C_GREEN)
            self._log("")
            self._log_styled("  ✅ 轨迹规划完成，请点击「任务开始」", "success")
            return
        a = self._traj_line.get_alpha()
        new_a = 0.0 if a>0.5 else 1.0
        self._traj_line.set_alpha(new_a)
        self._traj_line_3d.set_alpha(new_a)
        self._start_2d.set_alpha(new_a); self._end_2d.set_alpha(new_a)
        self._start_3d.set_alpha(new_a); self._end_3d.set_alpha(new_a)
        for ar in self._arrows: ar.set_alpha(new_a * 0.6)
        self._redraw()
        self._blink_count += 1
        self.after(500, self._do_blink)

    # === 阶段三 ===
    def _on_start(self):
        # 如果动画正在运行，点击则停止
        if self._anim_id is not None:
            self.after_cancel(self._anim_id)
            self._anim_id = None
            self._finish()
            return
        self.btn_env.configure(state="disabled")
        self.btn_plan.configure(state="disabled")
        self.btn_start.configure(text="⏹ 结束任务", fg_color=C_RED, hover_color="#dc2626")
        self._log_title("任务开始")
        self._log_styled("  正在加载 TD3 模型...", "info")
        def _worker():
            try:
                from Visualization_GUI.inference import load_model, run_follow
                model = load_model()
                self._log_styled("  模型加载成功，开始仿真推理...", "info")
                result = run_follow(model, self.generator.smooth_path)
                self.sim_result = result
                self._log_styled(f"  仿真完成: {result['steps']} 步，进度 {result['progress']:.0%}", "success")
                self.after(0, self._start_animation)
            except Exception as e:
                self.after(0, lambda: self._log_styled(f"  ❌ 错误: {e}", "error"))
                self.after(0, lambda: self.btn_start.configure(state="normal"))
        threading.Thread(target=_worker, daemon=True).start()

    def _start_animation(self):
        r = self.sim_result
        gen = self.generator; sp = gen.smooth_path
        # 隐藏之前的完整轨迹线，改用逐步绘制
        self._traj_line.set_alpha(0.0)
        self._traj_line_3d.set_alpha(0.0)
        for a in self._arrows: a.set_alpha(0.0)
        # 显示起点（终点等任务完成再显示）
        self._start_2d.set_alpha(1.0); self._end_2d.set_alpha(0.0)
        self._start_3d.set_alpha(1.0); self._end_3d.set_alpha(0.0)
        self.ax_2d.set_title("搜救任务执行中",fontsize=22)
        self.ax_3d.set_title("任务执行中 — 3D",fontsize=22)
        # 地面轨迹逐步绘制线
        self._ground_line, = self.ax_2d.plot([],[],color="green",linestyle="--",
            linewidth=3.5,label="地面搜救路径",zorder=3)
        self._ground_line_3d, = self.ax_3d.plot([],[],[],color="green",linestyle="--",
            linewidth=3.5,label="地面搜救路径",zorder=3)
        # 无人机轨迹逐步绘制线（半透明，不遮挡地面轨迹）
        self._uav_line, = self.ax_2d.plot([],[],color="#f97316",linewidth=2,alpha=0.6,label="无人机轨迹",zorder=4)
        self._gm, = self.ax_2d.plot([],[],"go",markersize=10,zorder=7)
        self._um, = self.ax_2d.plot([],[],marker="^",color="#f97316",markersize=12,zorder=7)
        self._dl, = self.ax_2d.plot([],[],"r-",linewidth=1,alpha=0.6,zorder=6)
        from matplotlib.lines import Line2D
        from matplotlib.patches import FancyArrow
        legend_handles = [
            Line2D([0],[0],color="green",linestyle="--",linewidth=2,label="地面搜救路径"),
            Line2D([0],[0],marker="o",color="w",markerfacecolor="green",markersize=10,label="起始点"),
            Line2D([0],[0],marker="o",color="w",markerfacecolor="red",markersize=10,label="终点"),
            Line2D([0],[0],color="#f97316",linewidth=2.5,label="无人机轨迹"),
        ]
        self.ax_2d.legend(handles=legend_handles, fontsize=19, loc="upper left")
        self._uav3d, = self.ax_3d.plot([],[],[],color="#f97316",linewidth=2,alpha=0.6,label="无人机轨迹",zorder=4)
        self.ax_3d.legend(handles=legend_handles, fontsize=19, loc="upper left")
        self._redraw()
        self._af=0; self._as=max(1,r["steps"]//60)
        self._aux=[]; self._auy=[]
        self._agx=[]; self._agy=[]  # 地面轨迹逐步点
        self._log_title("实时日志")
        self._log("")
        self._log_styled("  步数   地面坐标        无人机坐标      距离", "dim")
        self._log("")
        self._animate()

    def _animate(self):
        r = self.sim_result; total=r["steps"]; step=self._as
        end = min(self._af+step, total)

        # 地面轨迹：正常推进
        for i in range(self._af, end):
            gi = r["ground_indices"][i]
            gx_i,gy_i = r["ground_pts"][gi]
            self._agx.append(gx_i); self._agy.append(gy_i)

        # 无人机轨迹：延后 30 步，形成前后跟随效果
        uav_delay = 30
        uav_end = max(0, end - uav_delay)
        for i in range(max(0, self._af - uav_delay), uav_end):
            if i >= 0 and i < total:
                ux,uy = r["uav_pts"][i]
                self._aux.append(ux); self._auy.append(uy)

        self._af = end
        # 更新地面轨迹线
        self._ground_line.set_data(self._agx, self._agy)
        self._ground_line_3d.set_data_3d(self._agx, self._agy, [0]*len(self._agx))
        # 更新无人机轨迹线
        self._uav_line.set_data(self._aux, self._auy)
        gi = r["ground_indices"][end-1]
        gx,gy = r["ground_pts"][gi]
        # 无人机标记用延后的位置
        uav_idx = max(0, end - uav_delay - 1)
        ux,uy = r["uav_pts"][uav_idx]
        self._gm.set_data([gx],[gy]); self._um.set_data([ux],[uy])
        self._dl.set_data([gx,ux],[gy,uy])
        self._uav3d.set_data_3d(self._aux, self._auy, [20]*len(self._aux))
        # 指标用实际数据（不延后）
        dn=r["distances"][end-1]; ad=np.mean(r["distances"][:end])
        i5=np.mean(r["distances"][:end]<=5.0)*100; pr=end/total
        self.card_progress.set_value(f"{pr:.0%}")
        self.card_dist.set_value(f"{dn:.1f}m")
        self.card_avg.set_value(f"{ad:.1f}m")
        self.card_in5m.set_value(f"{i5:.0f}%")
        if end%(step*10)<step or end>=total:
            self._log_styled(
                f"  #{end:<5d}  G({gx:>4.0f},{gy:>4.0f})  U({ux:>4.0f},{uy:>4.0f})  Δ {dn:.1f}m",
                "data"
            )
        self._redraw()
        if end<total:
            self._anim_id = self.after(16, self._animate)
        else:
            # 补齐延后的无人机轨迹点
            for i in range(len(self._aux), total):
                ux,uy = r["uav_pts"][i]
                self._aux.append(ux); self._auy.append(uy)
            self._uav_line.set_data(self._aux, self._auy)
            self._uav3d.set_data_3d(self._aux, self._auy, [20]*len(self._aux))
            self._redraw()
            self._finish()

    def _finish(self):
        r = self.sim_result
        if r is None:
            return
        # 显示终点标记
        self._end_2d.set_alpha(1.0); self._end_3d.set_alpha(1.0)
        end = self._af if hasattr(self,'_af') else r["steps"]
        end = min(end, r["steps"])
        if end > 0:
            ad=np.mean(r["distances"][:end]); i5=np.mean(r["distances"][:end]<=5.0)*100
            md=np.max(r["distances"][:end])
        else:
            ad=0; i5=0; md=0
        self.ax_2d.set_title("搜救任务完成",fontsize=22)
        self.ax_3d.set_title("任务完成 — 3D",fontsize=22)
        self._redraw()
        self._log_title("任务完成")
        self._log_row("执行步数", str(end))
        self._log_row("完成进度", f"{end/r['steps']:.0%}")
        self._log_row("平均距离", f"{ad:.2f} m")
        self._log_row("最大距离", f"{md:.2f} m")
        self._log_row("5m内比例", f"{i5:.1f}%")
        self._log_sep()
        # 恢复按钮
        self.btn_start.configure(text="③ 任务开始", fg_color="#94a3b8", state="disabled")
        self.btn_env.configure(state="normal")
        self._anim_id = None

def main():
    app = TrajectoryVisualizerApp()
    app.mainloop()

if __name__ == "__main__":
    main()
