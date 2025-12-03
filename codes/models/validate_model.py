#import torch
#import torchreid
#from torchvision import transforms
#
## Configuración del dispositivo
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
## Cargar el modelo preentrenado de re-identificación
#model = torchreid.models.build_model(
#    name='resnet50',  # Modelo base (ResNet-50)
#    num_classes=12,   # Número de clases usado durante el entrenamiento
#    pretrained=False  # Evita usar un modelo base de clasificación ImageNet
#)
#checkpoint = torch.load('C:\\Users\\Matias\\Documents\\GitHub\\football_7_map_pass\\codes\\models\\model.pth.tar-300', map_location=device)
#model.load_state_dict(checkpoint['state_dict'])
#model = model.to(device)
#model.eval()  # Poner el modelo en modo de evaluación
#
## Preprocesamiento de imagen
#transform = transforms.Compose([
#    transforms.Resize((256, 128)),  # Tamaño estándar para ReID
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#])
#
## Crear una imagen de entrada simulada (dummy image)
#dummy_image = torch.randn(1, 3, 256, 128).to(device)  # Una imagen RGB simulada
#
## Pasar la imagen a través del modelo
#with torch.no_grad():  # Evitar cálculos de gradiente (modo evaluación)
#    embedding = model(dummy_image)
#
## Mostrar la dimensionalidad del embedding
#print(f"Dimensionalidad del embedding: {embedding.shape}")


import torch

# Ruta del archivo checkpoint
checkpoint_path = "C:\\Users\\Matias\\Documents\\GitHub\\football_7_map_pass\\codes\\models\\model.pth.tar-300"

# Cargar checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Cargar desde el archivo

# Inspeccionar las claves del checkpoint
print("Contenido del checkpoint:")
for key in checkpoint.keys():
    print(f"- {key}")

# Inspeccionar métricas, si están disponibles
if 'rank1' in checkpoint:
    print("\nRank-1 Accuracy:")
    print(f"Rank-1: {checkpoint['rank1']}")
else:
    print("\nNo se encontraron métricas como Rank-1.")

# Inspeccionar parámetros del modelo
if 'state_dict' in checkpoint:
    print("\nPesos del modelo (state_dict):")
    for param_key, param_value in checkpoint['state_dict'].items():
        print(f"{param_key}: {param_value.shape}")
else:
    print("\nNo se encontraron parámetros del modelo en el checkpoint.")
