from PIL import Image
import numpy as np

# 컬러 이미지 열기
img = Image.open('apple.jpg').convert('RGB')  # RGB로 변환

# 이미지 크기 확인 (필요시 모델 입력 크기에 맞게 리사이즈)
img = img.resize((300, 300))  # 예: 96x96 크기로 리사이즈

# numpy 배열로 변환
img_data = np.array(img)

# raw 바이너리로 저장 (픽셀은 RGB 3바이트씩 연속 저장됨)
with open('apple.raw', 'wb') as f:
    f.write(img_data.tobytes())
