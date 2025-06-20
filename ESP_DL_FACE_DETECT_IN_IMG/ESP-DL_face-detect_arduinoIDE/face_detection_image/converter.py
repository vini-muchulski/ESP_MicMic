from PIL import Image

# --- Configurações ---
INPUT_IMAGE_PATH = "ww_saul.jpg"       # Nome do seu arquivo de imagem de entrada
OUTPUT_HEADER_PATH = "image_to_detect.h"  # Nome do arquivo de saída .h
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
ARRAY_VARIABLE_NAME = "image_data"

# Abrir a imagem
try:
    img = Image.open(INPUT_IMAGE_PATH)
except FileNotFoundError:
    print(f"Erro: Arquivo de imagem '{INPUT_IMAGE_PATH}' não encontrado.")
    exit()

# Redimensionar e garantir que está no formato RGB
img = img.resize((TARGET_WIDTH, TARGET_HEIGHT))
img = img.convert("RGB")

# Extrair os dados brutos dos pixels
pixel_data = img.tobytes()

# Escrever o arquivo de cabeçalho .h
with open(OUTPUT_HEADER_PATH, "w") as f:
    f.write("#pragma once\n")
    f.write("#include <stdint.h>\n\n")
    f.write(f"#define IMAGE_WIDTH {TARGET_WIDTH}\n")
    f.write(f"#define IMAGE_HEIGHT {TARGET_HEIGHT}\n\n")
    f.write(f"const uint8_t {ARRAY_VARIABLE_NAME}[] = {{\n  ")
    
    # Escrever os bytes formatados
    count = 0
    for byte in pixel_data:
        f.write(f"0x{byte:02x}, ")
        count += 1
        if count % 16 == 0:  # Quebra de linha a cada 16 bytes para legibilidade
            f.write("\n  ")
    
    f.write("\n};\n")

print(f"Sucesso! Arquivo '{OUTPUT_HEADER_PATH}' gerado com sucesso.")
print(f"Tamanho do array: {len(pixel_data)} bytes.")