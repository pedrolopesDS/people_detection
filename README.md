# Detecção de Pessoas em Câmeras ao Vivo com YOLOv8

O projeto permite realizar a detecção de pessoas em vídeos transmitidos ao vivo por meio de câmeras públicas da cidade de Ourinhos-SP, utilizando o modelo YOLOv8.

---

## 🛠 Funcionalidades

- Seleção de câmeras ao vivo via menu interativo.
- Detecção de pessoas nos quadros capturados.
- Desenho de caixas delimitadoras ao redor das pessoas detectadas.
- Exibição da confiança da detecção diretamente no feed de vídeo.

---

## 🚀 Como executar o projeto

### 1. Pré-requisitos

**Não utilize o Google Colab.** Certifique-se de ter o Python 3.8 ou superior instalado e as bibliotecas necessárias. Para instalá-las, execute o seguinte comando no terminal:

```bash
pip install ultralytics opencv-python
```

### 2. Execute o código

1. Execute o script principal:
Execute o arquivo Python na sua IDE ou terminal

2. Escolha uma câmera no menu interativo digitando o número correspondente.

3. O feed de vídeo será exibido em uma janela, mostrando as detecções em tempo real.

4. Pressione `d` para encerrar o programa.

---

## 🔍 Explicação do Código

### 1. **Modelo YOLOv8**

O script utiliza o modelo YOLOv8n (versão "nano"), que é otimizado para velocidade e desempenho em hardware menos potente (meu caso).
```python
model = YOLO('yolov8n.pt')
```
No entanto, caso você tenha um hardware mais potente, você pode trocar para outros modelos mais robustos, os modelos existentes são:
1. small:
```python
model = YOLO('yolov8s.pt')  # Para o modelo small
```
2. medium:
```python
model = YOLO('yolov8m.pt')  # Para o modelo medium
```
3. large:
```python
model = YOLO('yolov8l.pt')  # Para o modelo large
```
3. extra-large:
```python
model = YOLO('yolov8x.pt')  # Para o modelo extra-large
```

### 2. **Configuração das câmeras**

As URLs das câmeras estão configuradas em um dicionário para fácil acesso e manutenção. Você pode adicionar ou remover câmeras diretamente neste dicionário.

```python
cameras = {
    "Calçadão": {
        "numero": 1,
        "nome": "Calçadão",
        "url": "https://..."
    },
    ...
}
```

### 3. **Escolha da câmera**

Exibe as opções disponíveis para o usuário e permite a seleção por meio do número associado a cada câmera:

```python
print("Escolha uma câmera:")
for nome_camera, info in cameras.items():
    print(f"{info['numero']} - {info['nome']}")

escolha = int(input("\nDigite o número da câmera: "))
selectcamera = next((info for info in cameras.values() if info["numero"] == escolha), None)
if selectcamera:
    url = selectcamera['url']
else:
    print("Câmera não encontrada.")
    exit()
```

### 4. **Configuração da captura de vídeo**

Usa a URL selecionada para abrir a transmissão ao vivo:

```python
capture = cv.VideoCapture(url)
if not capture.isOpened():
    print("Erro: Não foi possível abrir o vídeo.")
    exit()
```

### 5. **Processamento de vídeo**

Os quadros capturados são redimensionados para melhorar o desempenho:

```python
frame = cv.resize(frame, (640, 480))
```

O programa pula quadros com base em um contador, otimizando o uso de recursos:

```python
frame_skip = 2
frame_count += 1
if frame_count % frame_skip != 0:
    continue
```

### 6. **Inferência com YOLO**

Passa o quadro atual pelo modelo YOLO para realizar a detecção:

```python
results = model(frame)
```

### 7. **Iteração sobre as detecções**

Para cada detecção, são obtidos os valores necessários (coordenadas da caixa, classe e confiança):

```python
for result in results[0].boxes:
    box = result.xyxy[0].cpu().numpy()
    cls = int(result.cls[0])
    conf = result.conf[0]
```

Apenas pessoas (`cls == 0`) com confiança acima de 0.2 são processadas:

```python
if cls == 0 and conf > 0.2:
    x1, y1, x2, y2 = map(int, box)
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

### 8. **Exibição do vídeo processado**

Mostra os quadros processados em uma janela até que a tecla `d` seja pressionada:

```python
cv.imshow('Video - Detecção de Pessoas', frame)
if cv.waitKey(1) & 0xFF == ord('d'):
    break
```

### 9. **Encerramento do programa**

Libera os recursos utilizados e fecha todas as janelas abertas:

```python
capture.release()
cv.destroyAllWindows()
```

---

## 📌 Observações

- Não é possível executar no Google Colab
- Certifique-se de que as URLs das câmeras estão acessíveis, pode acontecer de alguma câmera esteja fora do ar.
- O modelo YOLOv8n foi escolhido pela eficiência em dispositivos de baixa capacidade, caso você tenha disponível um hardware melhor, escolha um modelo mais robuto para ter mais precisão.

---

## 🧩 Desafios
O YOLO requer um hardware robusto para esse tipo de aplicação. Por isso, foi necessário realizar otimizações para garantir seu funcionamento adequado. Além disso, a execução no Google Colab não foi viável, pois mesmo utilizando os outros Ambientes de Execução gratuitos, como 'T4 GPU' ou 'v2-8 TPU', a capacidade de processamento disponível não foi suficiente para atender às exigências do modelo.
