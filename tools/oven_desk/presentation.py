"""Recognition-first Tk shell around the existing capture and model interfaces."""
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
import cv2
from PIL import Image, ImageTk


class Presentation:
    def __init__(self,app,version):
        self.app=app;self.root=app.root;self.logs_open=tk.BooleanVar(value=False)
        self.result_name=tk.StringVar(value='尚未识别')
        self.feedback=tk.StringVar(value='打开图片，或选择并连接摄像头；再载入模型开始识别。')
        self.model_badge=tk.StringVar(value='○ 模型未就绪')
        self.source_badge=tk.StringVar(value='图像来源：未连接摄像头')
        self.camera_badge=tk.StringVar(value='○ 摄像头未连接')
        self.result_time=tk.StringVar();self.result_source=tk.StringVar();self.result_model=tk.StringVar()
        self._busy=False;self._preview_photo=None;self._result_photo=None;self._result_record=None
        style=app.style
        style.configure('TFrame',background='#F4F6F8')
        style.configure('TLabel',background='#F4F6F8',foreground='#1E293B',font=('Microsoft YaHei UI',10))
        style.configure('Card.TFrame',background='#FFFFFF')
        style.configure('Card.TLabel',background='#FFFFFF',foreground='#1E293B')
        style.configure('Muted.TLabel',foreground='#59697B',font=('Microsoft YaHei UI',9))
        style.configure('TButton',padding=(10,7),font=('Microsoft YaHei UI',10))
        style.configure('Accent.TButton',background='#2156B8',foreground='white',font=('Microsoft YaHei UI',11,'bold'),padding=(16,9))
        style.map('Accent.TButton',background=[('disabled','#ABB8C8'),('active','#174795')])
        style.configure('TNotebook.Tab',padding=(18,8))
        style.configure('Horizontal.TProgressbar',background='#2156B8',troughcolor='#E7EDF4')
        self.root.configure(bg='#F4F6F8')
        header=ttk.Frame(self.root,padding=(18,10,18,6));header.pack(fill='x')
        row=ttk.Frame(header);row.pack(fill='x')
        ttk.Label(row,text='HeT OvenCapture',font=('Microsoft YaHei UI',17,'bold')).pack(side='left')
        ttk.Label(row,text='  食材识别与图像采集',style='Muted.TLabel').pack(side='left',padx=8)
        ttk.Button(row,text='模型管理',command=lambda:self.show('模型管理')).pack(side='right')
        ttk.Label(row,text='v'+version.split('-')[0]+' · 实验版',style='Muted.TLabel').pack(side='right',padx=12)
        badges=ttk.Frame(header);badges.pack(fill='x',pady=(6,0))
        self.badges=[]
        for i,var in enumerate([self.model_badge,self.source_badge,self.camera_badge]):
            badges.columnconfigure(i,weight=1,uniform='badge')
            label=ttk.Label(badges,textvariable=var,style='Muted.TLabel',wraplength=340)
            label.grid(row=0,column=i,sticky='w',padx=(0,12));self.badges.append(label)
        app.tabs=ttk.Notebook(self.root);app.tabs.pack(fill='both',expand=True,padx=14,pady=6)
        self.pages={}
        home=ttk.Frame(app.tabs,padding=12);app.tabs.add(home,text='识别');self.pages['识别']=home
        self._home(home)
        capture=ttk.Frame(app.tabs,padding=8);app.tabs.add(capture,text='采集');self.pages['采集']=capture
        self._capture(capture)
        app.model_tab=app._scrollable_tab(app.tabs,'模型管理');self.pages['模型管理']=app.model_tab.master.master
        app._build_model_tab()
        self.model_details=ttk.Label(app.model_tab,text='选择模型并完成真实校验后显示身份信息。',wraplength=920,style='Muted.TLabel')
        self.model_details.pack(fill='x',pady=12)
        advanced=app._scrollable_tab(app.tabs,'高级设置');self.pages['高级设置']=advanced.master.master
        app._build_camera_advanced(advanced)
        ttk.Separator(advanced).pack(fill='x',pady=14)
        app.serial_tab=ttk.Frame(advanced);app.serial_tab.pack(fill='x');app._build_serial_tab()
        footer=ttk.Frame(self.root,padding=(16,0,16,10))
        # Reserve the error/status area before the expandable notebook at high DPI.
        footer.pack(side='bottom',before=app.tabs,fill='x')
        self.feedback_label=ttk.Label(footer,textvariable=self.feedback,wraplength=1100,font=('Microsoft YaHei UI',10),padding=(4,4))
        self.feedback_label.pack(fill='x')
        toggle=ttk.Checkbutton(footer,text='详细日志（本机）',variable=self.logs_open,command=self.toggle_logs)
        toggle.pack(anchor='w')
        self.log_panel=ttk.Frame(footer)
        app.log_box=tk.Text(self.log_panel,height=4,wrap='word',bg='#E9EEF4',fg='#263649',font=('Consolas',9),
                            borderwidth=0,padx=8,pady=6,state='disabled')
        scroll=ttk.Scrollbar(self.log_panel,orient='vertical',command=app.log_box.yview)
        app.log_box.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right',fill='y');app.log_box.pack(side='left',fill='both',expand=True)
        self.root.bind('<Configure>',self._resize,add='+')
        self.root.bind_all('<KeyPress-s>',app._key_save);self.root.bind_all('<KeyPress-S>',app._key_save)
        self.root.after_idle(self.refresh)

    def _home(self,parent):
        parent.columnconfigure(0,weight=3);parent.columnconfigure(1,weight=1,minsize=300)
        parent.rowconfigure(1,weight=1)
        source=ttk.Frame(parent);source.grid(row=0,column=0,columnspan=2,sticky='ew',pady=(0,10))
        ttk.Label(source,text='图像来源',font=('Microsoft YaHei UI',10,'bold')).pack(side='left',padx=(0,12))
        self.open_button=ttk.Button(source,text='打开图片…',command=self.app.choose_image);self.open_button.pack(side='left',padx=(0,8))
        ttk.Button(source,text='摄像头画面',command=self.app.use_camera_source).pack(side='left',padx=(0,8))
        ttk.Button(source,text='选择摄像头…',command=lambda:self.show('采集')).pack(side='left')
        image_panel=ttk.Frame(parent,style='Card.TFrame');image_panel.grid(row=1,column=0,sticky='nsew',padx=(0,12))
        self.image_label=tk.Label(image_panel,bg='#172431',fg='#D9E4EE',font=('Microsoft YaHei UI',13),
            text='还没有图像\n\n打开一张 JPG / PNG 图片\n或选择并连接摄像头',borderwidth=0)
        self.image_label.pack(fill='both',expand=True)
        self.image_label.bind('<Configure>',lambda event:self.draw_source())
        card=ttk.Frame(parent,style='Card.TFrame',padding=10);card.grid(row=1,column=1,sticky='nsew')
        self.result_card=card
        self.name_label=ttk.Label(card,textvariable=self.result_name,style='Card.TLabel',font=('Microsoft YaHei UI',20,'bold'))
        self.name_label.pack(anchor='w',fill='x',pady=(4,2))
        ttk.Separator(card).pack(fill='x',pady=3)
        ttk.Label(card,text='Top-3 模型分数',style='Card.TLabel',font=('Microsoft YaHei UI',8,'bold')).pack(anchor='w')
        self.score_rows=[]
        for i in range(3):
            group=ttk.Frame(card,style='Card.TFrame');group.pack(fill='x',pady=(3,0))
            name=tk.StringVar(value=f'{i+1}  —');score=tk.StringVar(value='—')
            ttk.Label(group,textvariable=name,style='Card.TLabel',font=('Microsoft YaHei UI',8)).pack(side='left')
            ttk.Label(group,textvariable=score,style='Card.TLabel',font=('Microsoft YaHei UI',8)).pack(side='right')
            self.score_rows.append((name,score))
        ttk.Separator(card).pack(fill='x',pady=3)
        ttk.Label(card,textvariable=self.result_time,style='Card.TLabel',font=('Microsoft YaHei UI',10,'bold'),wraplength=290).pack(anchor='w')
        self.scope_label=ttk.Label(card,text='识别后显示实际计时范围',style='Card.TLabel',foreground='#64748B',wraplength=420,font=('Microsoft YaHei UI',7))
        self.scope_label.pack(anchor='w',pady=(2,3))
        meta=ttk.Frame(card,style='Card.TFrame');meta.pack(fill='x',pady=(1,0))
        self.result_thumbnail=tk.Label(meta,bg='#EDF1F5',text='识别时图像',fg='#657487',height=1)
        self.result_thumbnail.pack(side='left',anchor='n',padx=(0,8))
        details=ttk.Frame(meta,style='Card.TFrame');details.pack(side='left',fill='x',expand=True)
        self.source_label=ttk.Label(details,textvariable=self.result_source,style='Card.TLabel',wraplength=290,font=('Microsoft YaHei UI',7))
        self.source_label.pack(anchor='w')
        self.model_label=ttk.Label(details,textvariable=self.result_model,style='Card.TLabel',foreground='#64748B',wraplength=290,font=('Microsoft YaHei UI',7))
        self.model_label.pack(anchor='w',pady=(3,0))
        card.bind('<Configure>',self._resize_result)
        actions=ttk.Frame(parent);actions.grid(row=2,column=0,columnspan=2,sticky='ew',pady=(12,0))
        self.recognize_button=ttk.Button(actions,text='识别当前画面',style='Accent.TButton',command=self.app.infer_current)
        self.recognize_button.pack(side='left',padx=(0,12))
        self.save_image_button=ttk.Button(actions,text='保存图片',command=self.app.save_display_image);self.save_image_button.pack(side='left',padx=(0,8))
        self.save_result_button=ttk.Button(actions,text='保存本次结果',command=self.app.save_recognition_result);self.save_result_button.pack(side='left')
        self.busy_text=ttk.Label(actions,text='',style='Muted.TLabel');self.busy_text.pack(side='right')
        self.progress=ttk.Progressbar(parent,mode='indeterminate')
        self.progress.grid(row=3,column=0,columnspan=2,sticky='ew',pady=(8,0))

    def _capture(self,parent):
        panes=ttk.Panedwindow(parent,orient='horizontal');panes.pack(fill='both',expand=True)
        left=ttk.Frame(panes,width=345);panes.add(left,weight=0)
        settings=ttk.Notebook(left);settings.pack(fill='both',expand=True)
        self.app.capture_tab=self.app._scrollable_tab(settings,'采集参数')
        self.app._build_capture_tab()
        right=ttk.Frame(panes,padding=(12,6));panes.add(right,weight=3)
        ttk.Label(right,textvariable=self.app.camera_status,wraplength=800).pack(anchor='w',pady=(0,8))
        self.app.preview=tk.Label(right,bg='#172431',fg='#D9E4EE',text='摄像头未连接\n\n请在左侧选择设备并明确打开',font=('Microsoft YaHei UI',12))
        self.app.preview.pack(fill='both',expand=True)
        bar=ttk.Frame(right);bar.pack(fill='x',pady=10)
        ttk.Button(bar,text='保存照片 (S)',style='Accent.TButton',command=self.app.save_photo).pack(side='left',padx=(0,8))
        self.app.burst_btn=ttk.Button(bar,text='开始连拍',command=self.app.toggle_burst);self.app.burst_btn.pack(side='left',padx=(0,8))
        self.app.record_btn=ttk.Button(bar,text='开始录像',command=self.app.toggle_recording);self.app.record_btn.pack(side='left',padx=(0,8))
        ttk.Button(bar,text='打开保存目录',command=self.app.open_output_folder).pack(side='left')
        ttk.Label(right,textvariable=self.app.photo_status,wraplength=750,style='Muted.TLabel').pack(anchor='w')

    def show(self,name):
        self.app.tabs.select(self.pages[name])

    def toggle_logs(self):
        if self.logs_open.get():self.log_panel.pack(fill='x',pady=(4,0))
        else:self.log_panel.pack_forget()

    def message(self,text,error=False):
        # Keep feedback visible without letting long labels/paths displace the result.
        # DesktopApp logs the complete message, and saved results retain full labels.
        self.feedback.set(self._fit(('错误：' if error else '')+str(text),max(500,self.root.winfo_width()-56),('Microsoft YaHei UI',10)))
        self.feedback_label.configure(foreground='#A32C30' if error else '#34465B')

    def clear_result(self):
        self._result_record=None
        self.result_name.set('尚未识别');self.result_time.set('')
        self.result_source.set('');self.result_model.set('')
        self.scope_label.configure(text='识别后显示实际计时范围')
        self.result_thumbnail.configure(image='',text='本次识别图像',height=1);self._result_photo=None
        for i,(name,score) in enumerate(self.score_rows):name.set(f'{i+1}  —');score.set('—')

    def show_result(self,record):
        self._result_record=record
        self.result_time.set(f"识别耗时  {record['latency_ms']:.1f} ms")
        self.scope_label.configure(text='含预处理、推理、输出检查及锁等待\n不含图片解码、加载、文件哈希、界面')
        rgb=Image.fromarray(cv2.cvtColor(record['frame'],cv2.COLOR_BGR2RGB));rgb.thumbnail((96,54),Image.Resampling.BILINEAR)
        self._result_photo=ImageTk.PhotoImage(rgb)
        self.result_thumbnail.configure(image=self._result_photo,text='',height=0)
        self._render_result_text()

    def _fit(self,text,width,font):
        text=str(text).replace('\n',' ');measure=tkfont.Font(root=self.root,font=font).measure
        if measure(text)<=width:return text
        low,high=0,len(text)
        while low<high:
            middle=(low+high+1)//2
            if measure(text[:middle]+'…')<=width:low=middle
            else:high=middle-1
        return text[:low]+'…'

    def _render_result_text(self):
        record=self._result_record
        if record is None:return
        width=min(max(240,self.result_card.winfo_width()-20),max(260,(self.root.winfo_width()-100)//3))
        result=record['prediction'];entries=result['top3'] if result['task']=='classification' else result['detections'][:3]
        title=entries[0]['label'] if result['task']=='classification' else f"检测到 {len(result['detections'])} 项" if entries else '无检测框'
        self.result_name.set(self._fit(title,width,('Microsoft YaHei UI',20,'bold')))
        for i,(name,score) in enumerate(self.score_rows):
            name.set(self._fit(f"{i+1}  {entries[i]['label']}",width-85,('Microsoft YaHei UI',8)) if i<len(entries) else f'{i+1}  —')
            score.set(f"{entries[i]['score']:.1%}" if i<len(entries) else '—')
        source=self._fit(record['source']['label'],max(160,width-110),('Microsoft YaHei UI',7))
        self.result_source.set(source+'\n'+record['finished_at'][:19].replace('T',' '))
        identity=record['model'];name=identity.get('catalog_id') or identity['display_name']
        self.result_model.set(self._fit(name[:8]+('…' if len(name)>8 else '')+' · '+identity['model_sha256'][:8],max(160,width-110),('Microsoft YaHei UI',7)))

    def draw_source(self):
        frame=self.app._recognition_frame()
        if frame is None:
            self.image_label.configure(image='',text='还没有图像\n\n打开一张 JPG / PNG 图片\n或选择并连接摄像头');self._preview_photo=None
            return
        width=self.image_label.winfo_width()-12;height=self.image_label.winfo_height()-12
        if width<30 or height<30:return
        h,w=frame.shape[:2];scale=min(width/w,height/h)
        size=(max(1,int(w*scale)),max(1,int(h*scale)))
        rgb=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize(size,Image.Resampling.BILINEAR)
        self._preview_photo=ImageTk.PhotoImage(rgb);self.image_label.configure(image=self._preview_photo,text='')

    def refresh(self):
        app=self.app;session=app.recognition;busy=app.model_loading or app.inference_running
        identity=getattr(app,'loaded_identity',None)
        badge_width=max(240,(self.root.winfo_width()-70)//3)
        self.model_badge.set(self._fit('● 模型就绪：'+identity['display_name'] if app.model is not None and identity else '○ 模型未就绪：请到模型管理载入',badge_width,('Microsoft YaHei UI',9)))
        source=session.source['label'];self.source_badge.set(self._fit('图像来源：'+source,badge_width,('Microsoft YaHei UI',9)))
        self.camera_badge.set('● 摄像头已连接' if app.camera_ready else '○ 摄像头未连接')
        ready=app.model is not None and self.app._recognition_frame() is not None and not busy
        self.recognize_button.configure(state='normal' if ready else 'disabled')
        self.save_image_button.configure(state='normal' if self.app._recognition_frame() is not None else 'disabled')
        self.save_result_button.configure(state='normal' if session.record is not None and not busy else 'disabled')
        self.busy_text.configure(text='正在加载模型…' if app.model_loading else '正在识别…' if app.inference_running else '')
        if busy!=self._busy:
            self.progress.start(15) if busy else self.progress.stop()
            self._busy=busy
        if identity and app.model is not None:
            self.model_details.configure(text=f"当前模型：{identity['display_name']}\n模型 SHA256：{identity['model_sha256']}\n类别文件 SHA256：{identity['classes_sha256']}\n运行模式：{identity['mode']} · 输入：{identity['input_shape']}\ncheckpoint：{identity.get('checkpoint_sha256') or '此格式未提供'}")
        else:self.model_details.configure(text='模型尚未就绪。选择和修改配置不会自动载入。')

    def _resize(self,event):
        if event.widget is not self.root:return
        wrap=max(240,(event.width-70)//3)
        for label in self.badges:label.configure(wraplength=wrap)
        self.feedback_label.configure(wraplength=max(500,event.width-48))
    def _resize_result(self,event):
        width=max(240,event.width-20)
        self.scope_label.configure(wraplength=width)
        for label in [self.source_label,self.model_label]:label.configure(wraplength=max(160,width-110))
        self._render_result_text()
