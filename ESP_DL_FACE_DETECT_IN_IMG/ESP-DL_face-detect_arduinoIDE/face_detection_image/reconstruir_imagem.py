from PIL import Image
import re

# --- Configurações ---
INPUT_HEADER_PATH = "image_to_detect.h"  # Arquivo .h gerado
OUTPUT_IMAGE_PATH = "reconstruida.png"  # Nome da imagem que será criada para verificação

# Variáveis para extrair do cabeçalho
width = 0
height = 0
pixel_data = []

print(f"Lendo o arquivo de cabeçalho: '{INPUT_HEADER_PATH}'")

try:
    with open(INPUT_HEADER_PATH, "r") as f:
        lines = f.readlines()

    for line in lines:
        # Extrai a largura e a altura das definições #define
        match_width = re.search(r'#define\s+IMAGE_WIDTH\s+(\d+)', line)
        if match_width:
            width = int(match_width.group(1))
            print(f"Largura encontrada: {width}")

        match_height = re.search(r'#define\s+IMAGE_HEIGHT\s+(\d+)', line)
        if match_height:
            height = int(match_height.group(1))
            print(f"Altura encontrada: {height}")

        # Extrai os bytes hexadecimais do array
        # Esta expressão regular encontra todos os números hexadecimais (0x..) na linha
        hex_values = re.findall(r'0x([0-9a-fA-F]{2})', line)
        if hex_values:
            # Converte os valores hexadecimais (string) para inteiros e os adiciona à lista
            pixel_data.extend([int(val, 16) for val in hex_values])

    if not width or not height:
        raise ValueError("Não foi possível encontrar as definições de IMAGE_WIDTH ou IMAGE_HEIGHT no arquivo.")
    
    if not pixel_data:
        raise ValueError("Nenhum dado de pixel foi encontrado no arquivo.")

    print(f"Total de bytes de pixel lidos: {len(pixel_data)}")
    
    # Verifica se a quantidade de dados corresponde às dimensões (para RGB888, são 3 bytes por pixel)
    expected_size = width * height * 3
    if len(pixel_data) != expected_size:
        print(f"Aviso: O tamanho dos dados ({len(pixel_data)}) não corresponde ao esperado ({expected_size}). A imagem pode ficar distorcida.")

    # Converte a lista de bytes para um objeto de bytes
    pixel_bytes = bytes(pixel_data)
    
    # Cria uma nova imagem a partir dos dados brutos
    # O modo 'RGB' informa à Pillow que os dados são sequências de 3 bytes (Vermelho, Verde, Azul)
    img = Image.frombytes('RGB', (width, height), pixel_bytes)

    # Salva a imagem reconstruída
    img.save(OUTPUT_IMAGE_PATH)

    print(f"\nSucesso! Imagem reconstruída e salva como '{OUTPUT_IMAGE_PATH}'.")
    print("Abra este arquivo para verificar se a conversão está correta.")

except FileNotFoundError:
    print(f"Erro: Arquivo '{INPUT_HEADER_PATH}' não encontrado. Certifique-se de que ele está na mesma pasta que este script.")
except Exception as e:
    print(f"Ocorreu um erro: {e}")