import cv2
from ultralytics import YOLO

PATH = ''

# Caminhos dos modelos e das imagens
models = [
    f"{PATH}/500/best.pt",
    f"{PATH}/3000/best.pt",
    f"{PATH}/5000/best.pt",
]

images = {
    "homem": f"{PATH}/img_homem.jpg",
    "mulher": f"{PATH}/img_mulher.jpg",
    "varios": f"{PATH}/img_varios.webp",
}

output_dir = f"{PATH}/results/"

# Função para processar as imagens
def process_image(image_path, model_path, output_path):
    # Carregar o modelo YOLO
    model = YOLO(model_path)

    # Carregar e redimensionar a imagem para 640x640
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (640, 640))

    # Realizar a predição
    results = model.predict(resized_image)

    # Obter os resultados e desenhar as bounding boxes
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]}: {score:.2f}"
            cv2.rectangle(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Salvar a imagem com bounding boxes
    cv2.imwrite(output_path, resized_image)

# Processar cada imagem com cada modelo
for model_path in models:
    model_name = model_path.split("/")[-2]  # Extrair o nome do modelo (500, 3000 ou 5000)
    for image_label, image_path in images.items():
        output_path = f"{output_dir}{model_name}_{image_label}_detected.jpg"
        print(image_path)
        print(model_path)
        process_image(image_path, model_path, output_path)
        print(f"Resultado salvo: {output_path}")
