import tkinter as tk
from utils.getkeys import key_check
import keyboard
import cv2
from utils.grabscreen import grab_screen, screen_mask, pixel_filter
import torch
from torchvision import transforms, models
import time

# 加载模型
model = models.resnet18() 
model.load_state_dict(torch.load("./pkl/trained28_large_50.pth")) 
model.eval()
print("Loaded model")

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
])

# 预测函数
def predict(image):
    image = preprocess(image)
    image = image.unsqueeze(0) 
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()

# 间隔
sleepy = 0.1

# 窗口设置
class PredictionWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes("-topmost", True) 
        self.root.overrideredirect(True)
        self.root.geometry("200x80+500+150")

        # 窗口内容
        self.label = tk.Label(self.root, text="...", font=("Helvetica", 12))
        self.label.pack(expand=True)

        self.start_x = 0
        self.start_y = 0

    def start_drag(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def do_drag(self, event):
        x = self.root.winfo_x() + event.x - self.start_x
        y = self.root.winfo_y() + event.y - self.start_y
        self.root.geometry(f"+{x}+{y}")

    def update_label(self, text):
        self.label.config(text=text)
        self.root.update()

# 创建窗口
prediction_window = PredictionWindow()

# 开始
keyboard.wait('B')
time.sleep(sleepy)
keyboard.press('Q')

while True:
    # 图像预处理
    image = grab_screen(region=(270, 600, 2286, 1299))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = screen_mask(image, region=(270, 600, 2286, 1299))
    image = pixel_filter(image)
    image = cv2.resize(image, (224, 224))

    action = predict(image)  # 获取预测结果

    # 根据预测采取动作
    if action == 0:
        print(f"Down")
        keyboard.press("s")
        keyboard.release("a")
        keyboard.release("d")
        time.sleep(sleepy)
        keyboard.release("s")
    elif action == 1:
        keyboard.release("s")
        keyboard.release("a")
        keyboard.release("d")
        time.sleep(sleepy)
    elif action == 2:
        print(f"Left")
        keyboard.press("a")
        keyboard.release("d")
        keyboard.release("s")
        time.sleep(sleepy)
        keyboard.release("a")
    elif action == 3:
        print(f"Right")
        keyboard.press("d")
        keyboard.release("a")
        keyboard.release("s")
        time.sleep(sleepy)
        keyboard.release("d")

    # 结束模拟
    keys = key_check()
    if keys == "H":
        break

    # 可视化
    cv2.imshow("input image", cv2.resize(image, (400, 400)))
    cv2.setWindowProperty("input image", cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(1)

    # 更新窗口显示预测结果
    prediction_window.update_label(f"Prediction: {action}")

keyboard.release('Q')
prediction_window.root.destroy()
cv2.destroyAllWindows()
