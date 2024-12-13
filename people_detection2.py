import cv2 as cv
from ultralytics import YOLO

cameras = {
    "Calçadão": {
        "numero": 1,
        "nome": "Calçadão",
        "url": "https://ipservice.aovivonanet.com.br:4433/hls/ourinhos-calcadao/index.m3u8"
    },
    "Calçadão 2": {
        "numero": 2,
        "nome": "Calçadão 2",
        "url": "https://ipservice.aovivonanet.com.br:4433/hls/ourinhos-calcadao-2/index.m3u8"
    },
    "Praça Recanto dos Pássaros": {
        "numero": 3,
        "nome": "Praça Recanto dos Pássaros",
        "url": "https://ipservice.aovivonanet.com.br:4433/hls/ourinhos-praca-recanto-passaros/index.m3u8"
    },
    "Plaza Shopping": {
        "numero": 4,
        "nome": "Plaza Shopping",
        "url": "https://ipservice.aovivonanet.com.br:4433/hls/ourinhos-plaza-shopping/index.m3u8"
    },
    "Feira da lua Cohab": {
        "numero": 5,
        "nome": "Feira da lua Cohab",
        "url": "https://ipservice.aovivonanet.com.br:4433/hls/ourinhos-feira-da-lua-cohab/index.m3u8"
    },
    "Praça dos Skatistas": {
        "numero": 6,
        "nome": "Praça dos Skatistas",
        "url": "https://ipservice.aovivonanet.com.br:4433/hls/ourinhos-praca-dos-shatistas/index.m3u8"
    },
    "Praça Kennedy": {
        "numero": 7,
        "nome": "Praça Kennedy",
        "url": "https://ipservice.aovivonanet.com.br:4433/hls/ourinhos-praca-kennedy/index.m3u8"
    },
    "Praça dos Ferroviários": {
        "numero": 8,
        "nome": "Praça dos Ferroviários",
        "url": "https://ipservice.aovivonanet.com.br:4433/hls/ourinhos-praca-dos-ferroviarios/index.m3u8"
    },
    "Praça São Pio X": {
        "numero": 9,
        "nome": "Praça São Pio X",
        "url": "https://ipservice.aovivonanet.com.br:4433/hls/ourinhos-sao-pio-x/index.m3u8"
    },
    "Praça Vila Perino": {
        "numero": 10,
        "nome": "Praça Vila Perino",
        "url": "https://ipservice.aovivonanet.com.br:4433/hls/ourinhos-praca-vila-peino/index.m3u8"
    },
    "Rio de Janeiro x Antônio Carlos Mori": {
        "numero": 11,
        "nome": "Rio de Janeiro x Antônio Carlos Mori",
        "url": "https://ipservice.aovivonanet.com.br:4433/hls/ourinhos-rio-de-janeiro-x-antonio-mori/index.m3u8"
    },
    "Altino Arantes x Cardoso Ribeiro": {
        "numero": 12,
        "nome": "Altino Arantes x Cardoso Ribeiro",
        "url": "https://ipservice.aovivonanet.com.br:4433/hls/ourinhos-altino-arantes-x-cardoso-ribeiro/index.m3u8"
    }
}

model = YOLO("yolov8n.pt")

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

capture = cv.VideoCapture(url)
if not capture.isOpened():
    print("Erro: Não foi possível abrir o vídeo.")
    exit()

frame_skip = 2
frame_count = 0

while True:
    frame_count += 1
    isTrue, frame = capture.read()
    if not isTrue:
        print("Erro: Não foi possível ler o quadro.")
        break

    if frame_count % frame_skip != 0:
        continue

    frame = cv.resize(frame, (640, 480))

    results = model(frame)
    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy()
        cls = int(result.cls[0])
        conf = result.conf[0]

        if cls == 0 and conf > 0.2:
            x1, y1, x2, y2 = map(int, box)
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv.imshow('Video - Detecção de Pessoas', frame)
    if cv.waitKey(1) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
