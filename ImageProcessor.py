import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from sklearn.cluster import KMeans
import cv2
import warnings

# 콘솔 경고 메시지 숨기기
warnings.filterwarnings('ignore')

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry("1100x600")
        
        # 공통 변수들
        self.original_image = None
        self.processed_image = None
        self.display_ratio = 1.0
        
        # iOS 스타일 색상 테마
        self.bg_color = "#F2F2F7"
        self.card_color = "#FFFFFF"
        self.accent_color = "#007AFF"
        self.text_color = "#000000"
        self.secondary_text = "#8E8E93"
        
        self.configure_style()
        self.setup_ui()
    
    def configure_style(self):
        """iOS 스타일 테마 설정"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 배경색
        self.root.configure(bg=self.bg_color)
        
        # Frame 스타일
        style.configure('Card.TFrame', background=self.card_color, relief='flat')
        style.configure('TFrame', background=self.bg_color)
        
        # Label 스타일
        style.configure('TLabel', background=self.card_color, foreground=self.text_color, 
                       font=('SF Pro Display', 11))
        style.configure('Title.TLabel', background=self.bg_color, foreground=self.text_color,
                       font=('SF Pro Display', 16, 'bold'))
        style.configure('Secondary.TLabel', background=self.card_color, foreground=self.secondary_text,
                       font=('SF Pro Display', 10))
        
        # Button 스타일
        style.configure('Accent.TButton', font=('SF Pro Display', 11), padding=(20, 10))
        style.map('Accent.TButton',
                 background=[('active', self.accent_color), ('!active', self.accent_color)],
                 foreground=[('active', 'white'), ('!active', 'white')])
        
        # Notebook 스타일
        style.configure('TNotebook', background=self.bg_color, borderwidth=0)
        style.configure('TNotebook.Tab', padding=[20, 10], font=('SF Pro Display', 11))
        
    def setup_ui(self):
        # 메인 프레임
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 공통 컨트롤 프레임 (카드 스타일)
        control_frame = tk.Frame(main_frame, bg=self.card_color, relief='flat', bd=0)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 내부 패딩을 위한 프레임
        control_inner = tk.Frame(control_frame, bg=self.card_color)
        control_inner.pack(padx=15, pady=8)
        
        # 버튼들
        load_btn = tk.Button(control_inner, text="Load Image", 
                            command=self.load_image,
                            bg=self.accent_color, fg='white',
                            font=('SF Pro Display', 10), 
                            relief='flat', bd=0,
                            padx=20, pady=8, cursor='hand2')
        load_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        save_btn = tk.Button(control_inner, text="Save Image", 
                            command=self.save_image,
                            bg='#E5E5EA', fg=self.text_color,
                            font=('SF Pro Display', 10), 
                            relief='flat', bd=0,
                            padx=20, pady=8, cursor='hand2')
        save_btn.pack(side=tk.LEFT)
        
        # 탭 생성
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 탭 1: 안티앨리어싱 제거
        self.antialiasing_tab = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(self.antialiasing_tab, text="Anti-Aliasing Removal")
        self.setup_antialiasing_tab()
        
        # 탭 2: 색상 변경
        self.color_change_tab = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(self.color_change_tab, text="Color Change")
        self.setup_color_change_tab()
    
    def setup_antialiasing_tab(self):
        """안티앨리어싱 제거 탭 설정"""
        # 컨트롤 카드
        control_card = tk.Frame(self.antialiasing_tab, bg=self.card_color, relief='flat')
        control_card.pack(fill=tk.X, padx=10, pady=(5, 0))
        
        control_inner = tk.Frame(control_card, bg=self.card_color)
        control_inner.pack(padx=12, pady=8)
        
        # 색상 개수
        tk.Label(control_inner, text="Color Count", bg=self.card_color, 
                fg=self.text_color, font=('SF Pro Display', 10)).grid(row=0, column=0, padx=(0, 8))
        
        self.color_count = tk.StringVar(value="8")
        color_spinbox = tk.Spinbox(control_inner, from_=2, to=256, width=6, 
                                   textvariable=self.color_count,
                                   font=('SF Pro Display', 10),
                                   relief='flat', bd=1,
                                   highlightthickness=1, highlightbackground='#E5E5EA')
        color_spinbox.grid(row=0, column=1, padx=(0, 15))
        
        # 처리 방법
        tk.Label(control_inner, text="Method", bg=self.card_color,
                fg=self.text_color, font=('SF Pro Display', 10)).grid(row=0, column=2, padx=(0, 8))
        
        self.method = tk.StringVar(value="kmeans")
        method_menu = ttk.Combobox(control_inner, textvariable=self.method,
                                  values=["kmeans", "quantize", "threshold"],
                                  state="readonly", width=10, font=('SF Pro Display', 10))
        method_menu.grid(row=0, column=3, padx=(0, 15))
        
        # 처리 버튼
        process_btn = tk.Button(control_inner, text="Process", 
                               command=self.process_antialiasing,
                               bg=self.accent_color, fg='white',
                               font=('SF Pro Display', 10),
                               relief='flat', bd=0,
                               padx=20, pady=6, cursor='hand2')
        process_btn.grid(row=0, column=4)
        
        # 이미지 디스플레이 영역
        image_container = tk.Frame(self.antialiasing_tab, bg=self.bg_color)
        image_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        
        # 원본 이미지 카드
        original_card = tk.Frame(image_container, bg=self.card_color, relief='flat')
        original_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        tk.Label(original_card, text="Original", bg=self.card_color,
                fg=self.secondary_text, font=('SF Pro Display', 9)).pack(pady=(8, 5))
        
        self.aa_original_label = tk.Label(original_card, text="Load an image to start",
                                         bg=self.card_color, fg=self.secondary_text,
                                         font=('SF Pro Display', 9))
        self.aa_original_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # 처리된 이미지 카드
        processed_card = tk.Frame(image_container, bg=self.card_color, relief='flat')
        processed_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        tk.Label(processed_card, text="Processed", bg=self.card_color,
                fg=self.secondary_text, font=('SF Pro Display', 9)).pack(pady=(8, 5))
        
        self.aa_processed_label = tk.Label(processed_card, text="Processed result will appear here",
                                          bg=self.card_color, fg=self.secondary_text,
                                          font=('SF Pro Display', 9))
        self.aa_processed_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
    
    def setup_color_change_tab(self):
        """색상 변경 탭 설정"""
        # 메인 컨테이너 - 가로 레이아웃
        main_container = tk.Frame(self.color_change_tab, bg=self.bg_color)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 왼쪽: 캔버스 카드
        canvas_card = tk.Frame(main_container, bg=self.card_color, relief='flat')
        canvas_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        canvas_inner = tk.Frame(canvas_card, bg=self.card_color)
        canvas_inner.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # 캔버스
        self.canvas = tk.Canvas(canvas_inner, width=650, height=400, 
                               bg="#FAFAFA", highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # 안내 라벨
        info_label = tk.Label(canvas_inner, 
                             text="Click to pick color",
                             bg=self.card_color, fg=self.secondary_text,
                             font=('SF Pro Display', 8))
        info_label.pack(pady=(5, 0))
        
        # 오른쪽: 색상 설정 카드
        color_card = tk.Frame(main_container, bg=self.card_color, relief='flat')
        color_card.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0))
        
        color_inner = tk.Frame(color_card, bg=self.card_color)
        color_inner.pack(padx=12, pady=10)
        
        # 목표 색상
        tk.Label(color_inner, text="Target Color", bg=self.card_color,
                fg=self.text_color, font=('SF Pro Display', 10, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 8))
        
        self.create_color_inputs(color_inner, 'target', 1)
        
        # 구분선
        tk.Frame(color_inner, height=1, bg='#E5E5EA').grid(row=5, column=0, columnspan=2, sticky='ew', pady=10)
        
        # 새 색상
        tk.Label(color_inner, text="New Color", bg=self.card_color,
                fg=self.text_color, font=('SF Pro Display', 10, 'bold')).grid(row=6, column=0, columnspan=2, pady=(0, 8))
        
        self.create_color_inputs(color_inner, 'new', 7)
        
        # 빠른 선택 버튼들
        quick_frame = tk.Frame(color_inner, bg=self.card_color)
        quick_frame.grid(row=11, column=0, columnspan=2, pady=(10, 0))
        
        tk.Label(quick_frame, text="Quick", bg=self.card_color,
                fg=self.secondary_text, font=('SF Pro Display', 8)).pack(pady=(0, 5))
        
        btn_frame = tk.Frame(quick_frame, bg=self.card_color)
        btn_frame.pack()
        
        # 흰색
        tk.Button(btn_frame, bg='#FFFFFF', width=3, height=1,
                 relief='solid', bd=1, cursor='hand2',
                 command=lambda: self.set_new_color(255, 255, 255, 255)).pack(side=tk.LEFT, padx=3)
        
        # 검은색
        tk.Button(btn_frame, bg='#000000', width=3, height=1,
                 relief='solid', bd=1, cursor='hand2',
                 command=lambda: self.set_new_color(0, 0, 0, 255)).pack(side=tk.LEFT, padx=3)
        
        # More 버튼
        tk.Button(btn_frame, text="More", bg='#E5E5EA',
                 fg=self.text_color, font=('SF Pro Display', 8),
                 relief='flat', bd=0, width=6, height=1,
                 cursor='hand2', command=self.open_color_palette).pack(side=tk.LEFT, padx=3)
        
        # 적용 버튼
        apply_btn = tk.Button(color_inner, text="Apply", 
                             command=self.change_color,
                             bg=self.accent_color, fg='white',
                             font=('SF Pro Display', 10, 'bold'),
                             relief='flat', bd=0,
                             padx=30, pady=8, cursor='hand2')
        apply_btn.grid(row=12, column=0, columnspan=2, pady=(15, 0))
    
    def create_color_inputs(self, parent, prefix, start_row):
        """RGBA 입력 필드 생성"""
        labels = ['R', 'G', 'B', 'A']
        entries = []
        
        for i, label in enumerate(labels):
            row = start_row + i
            
            tk.Label(parent, text=label, bg=self.card_color,
                    fg=self.text_color, font=('SF Pro Display', 9),
                    width=2).grid(row=row, column=0, sticky='e', padx=(0, 4), pady=2)
            
            entry = tk.Entry(parent, width=8, font=('SF Pro Display', 9),
                           relief='flat', bd=1, highlightthickness=1,
                           highlightbackground='#E5E5EA', highlightcolor=self.accent_color)
            entry.grid(row=row, column=1, pady=2)
            
            if label == 'A':
                entry.insert(0, '255')
            
            entries.append(entry)
            
            # 속성으로 저장
            setattr(self, f'{prefix}_{label.lower()}', entry)
        
        return entries
    
    def set_new_color(self, r, g, b, a):
        """새 색상 값 설정"""
        self.new_r.delete(0, tk.END)
        self.new_r.insert(0, str(r))
        
        self.new_g.delete(0, tk.END)
        self.new_g.insert(0, str(g))
        
        self.new_b.delete(0, tk.END)
        self.new_b.insert(0, str(b))
        
        self.new_a.delete(0, tk.END)
        self.new_a.insert(0, str(a))
    
    def open_color_palette(self):
        """색상 팔레트 모달 열기"""
        color = colorchooser.askcolor(title="Choose Color")
        if color[0]:  # RGB 튜플이 반환됨
            r, g, b = [int(c) for c in color[0]]
            self.set_new_color(r, g, b, 255)
    
    def load_image(self):
        """이미지 파일을 불러옵니다."""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # 원본 이미지와 처리된 이미지 모두 초기화
                self.original_image = Image.open(file_path).convert('RGBA')
                self.processed_image = None
                
                # 모든 디스플레이 업데이트
                self.update_displays()
                
                # 처리된 이미지 라벨 초기화
                self.aa_processed_label.config(image='', text="Processed result will appear here",
                                              fg=self.secondary_text)
                
                messagebox.showinfo("Success", "Image loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def update_displays(self):
        """모든 디스플레이를 업데이트합니다."""
        if self.original_image:
            # 안티앨리어싱 탭 - 원본 이미지 업데이트
            display_img = self.resize_for_display(self.original_image, 350)
            photo = ImageTk.PhotoImage(display_img)
            self.aa_original_label.config(image=photo, text="", bg=self.card_color)
            self.aa_original_label.image = photo
            
            # 색상 변경 탭 - 캔버스 업데이트
            self.update_canvas()
    
    def resize_for_display(self, image, max_size=350):
        """이미지를 디스플레이용으로 크기 조정합니다."""
        width, height = image.size
        if width > max_size or height > max_size:
            ratio = min(max_size/width, max_size/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image
    
    def update_canvas(self):
        """캔버스에 이미지를 표시합니다."""
        if self.processed_image:
            img = self.processed_image
        elif self.original_image:
            img = self.original_image
        else:
            return
        
        canvas_w = 650
        canvas_h = 400
        img_w, img_h = img.size
        
        # 캔버스에 맞게 축소 비율 계산
        self.display_ratio = min(canvas_w / img_w, canvas_h / img_h)
        
        if self.display_ratio < 1.0:
            new_w = int(img_w * self.display_ratio)
            new_h = int(img_h * self.display_ratio)
            display_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            self.display_ratio = 1.0
            display_img = img.copy()
        
        img_tk = ImageTk.PhotoImage(display_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.img = img_tk
    
    def on_canvas_click(self, event):
        """캔버스 클릭 시 스포이드 기능"""
        img = self.processed_image if self.processed_image else self.original_image
        if img is None:
            return
        
        # 화면 좌표를 원본 이미지 좌표로 변환
        orig_x = int(event.x / self.display_ratio)
        orig_y = int(event.y / self.display_ratio)
        
        if 0 <= orig_x < img.width and 0 <= orig_y < img.height:
            pixel = img.getpixel((orig_x, orig_y))
            
            # 목표 색상 입력창에 채우기
            self.target_r.delete(0, tk.END)
            self.target_r.insert(0, str(pixel[0]))
            
            self.target_g.delete(0, tk.END)
            self.target_g.insert(0, str(pixel[1]))
            
            self.target_b.delete(0, tk.END)
            self.target_b.insert(0, str(pixel[2]))
            
            self.target_a.delete(0, tk.END)
            self.target_a.insert(0, str(pixel[3]))
    
    def process_antialiasing(self):
        """안티앨리어싱 제거 처리"""
        if not self.original_image:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        
        try:
            color_count = int(self.color_count.get())
            method = self.method.get()
            
            # RGB로 변환하여 처리
            img_array = np.array(self.original_image.convert('RGB'))
            
            if method == "kmeans":
                processed_array = self.process_with_kmeans(img_array, color_count)
            elif method == "quantize":
                processed_array = self.process_with_quantize(img_array, color_count)
            elif method == "threshold":
                processed_array = self.process_with_threshold(img_array, color_count)
            
            # 알파 채널 보존
            if self.original_image.mode == 'RGBA':
                alpha = np.array(self.original_image)[:, :, 3]
                processed_rgba = np.dstack((processed_array, alpha))
                self.processed_image = Image.fromarray(processed_rgba.astype(np.uint8), 'RGBA')
            else:
                self.processed_image = Image.fromarray(processed_array.astype(np.uint8), 'RGB')
            
            # 처리된 이미지 표시
            display_img = self.resize_for_display(self.processed_image, 350)
            photo = ImageTk.PhotoImage(display_img)
            self.aa_processed_label.config(image=photo, text="", bg=self.card_color)
            self.aa_processed_label.image = photo
            
            # 색상 변경 탭의 캔버스도 업데이트
            self.update_canvas()
            
            messagebox.showinfo("Success", "Image processing completed!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during image processing: {str(e)}")
    
    def process_with_kmeans(self, img_array, color_count):
        """K-means 클러스터링을 사용한 색상 양자화"""
        pixels = img_array.reshape(-1, 3)
        kmeans = KMeans(n_clusters=color_count, random_state=42, n_init=10)
        kmeans.fit(pixels)
        new_pixels = kmeans.cluster_centers_[kmeans.labels_]
        return new_pixels.reshape(img_array.shape)
    
    def process_with_quantize(self, img_array, color_count):
        """PIL의 quantize를 사용한 색상 양자화"""
        pil_image = Image.fromarray(img_array)
        quantized = pil_image.quantize(colors=color_count)
        return np.array(quantized.convert('RGB'))
    
    def process_with_threshold(self, img_array, color_count):
        """단순 임계값을 사용한 색상 양자화"""
        step = 256 // color_count
        processed = img_array.copy().astype(float)
        processed = (processed // step) * step
        processed = np.clip(processed, 0, 255)
        return processed
    
    def change_color(self):
        """특정 색상을 다른 색상으로 변경"""
        # 처리된 이미지가 있으면 그것을 사용, 없으면 원본 사용
        img = self.processed_image if self.processed_image else self.original_image
        
        if img is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        
        try:
            target_color = (
                int(self.target_r.get()), 
                int(self.target_g.get()), 
                int(self.target_b.get()), 
                int(self.target_a.get())
            )
            new_color = (
                int(self.new_r.get()), 
                int(self.new_g.get()), 
                int(self.new_b.get()), 
                int(self.new_a.get())
            )
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for all color values.")
            return
        
        # 이미지 데이터 변경
        img_data = img.getdata()
        new_img_data = []
        
        for item in img_data:
            if item == target_color:
                new_img_data.append(new_color)
            else:
                new_img_data.append(item)
        
        # 새 이미지 생성
        new_img = Image.new('RGBA', img.size)
        new_img.putdata(new_img_data)
        self.processed_image = new_img
        
        # 색상 변경 탭 캔버스 업데이트
        self.update_canvas()
        
        # 안티앨리어싱 탭의 원본 이미지 위치에도 표시
        display_img = self.resize_for_display(self.processed_image, 350)
        photo = ImageTk.PhotoImage(display_img)
        self.aa_original_label.config(image=photo, text="", bg=self.card_color)
        self.aa_original_label.image = photo
        
        messagebox.showinfo("Success", "Color changed successfully!")
    
    def save_image(self):
        """처리된 이미지를 저장합니다."""
        img_to_save = self.processed_image if self.processed_image else self.original_image
        
        if not img_to_save:
            messagebox.showwarning("Warning", "No image to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=".png",
            filetypes=[
                ("PNG Files", "*.png"),
                ("JPEG Files", "*.jpg"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            try:
                img_to_save.save(file_path)
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving image: {str(e)}")

def main():
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()

if __name__ == "__main__":
    main()
