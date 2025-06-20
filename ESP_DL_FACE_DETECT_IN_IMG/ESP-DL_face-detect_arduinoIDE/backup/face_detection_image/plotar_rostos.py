import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# aqui você só precisa editar estes valores:
face_boxes = [
   ( 5, 245, 235, 246)
]

# carregue sua imagem (altere o caminho conforme necessário)
img = Image.open('reconstruida.png')

fig, ax = plt.subplots()
ax.imshow(img)

for x, y, w, h in face_boxes:
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

ax.axis('off')
plt.show()
