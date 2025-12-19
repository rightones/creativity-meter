# Creativity Quantification with Diffusion Inversion

## 사용법

### 단일 이미지 측정

```bash
python main.py measure --image path/to/your/image.jpg
```

출력 예시:
```
------------------------------
Original: path/to/your/image.jpg
Similarity: 0.8234
Creativity Score: 0.1766
------------------------------
```

### 폴더 내 이미지 일괄 측정

```bash
python main.py measure --dir path/to/images/ --output results.csv
```

### 재구성 이미지 저장

```bash
python main.py measure --image image.jpg --save_recon ./reconstructed/
```