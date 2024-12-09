from ultralytics import YOLO
from PIL import Image
from pathlib import Path

def diviser_image_en_blocs(image_path, taille_bloc, output_dir):
    """
    Divise une grande image en blocs de taille spécifiée.

    :param image_path: Chemin de l'image à diviser.
    :param taille_bloc: Taille des blocs (largeur, hauteur).
    :param output_dir: Répertoire de sortie pour enregistrer les blocs.
    """
    # Charger l'image
    image = Image.open(image_path)
    largeur, hauteur = image.size
    bloc_largeur, bloc_hauteur = taille_bloc

    # Diviser l'image en blocs
    for y in range(0, hauteur, bloc_hauteur):
        for x in range(0, largeur, bloc_largeur):
            # Définir les limites du bloc
            box = (x, y, min(x + bloc_largeur, largeur), min(y + bloc_hauteur, hauteur))
            bloc = image.crop(box)
            
            # Nom du fichier pour chaque bloc
            bloc_nom = f"bloc_{x}_{y}.png"
            bloc_chemin = f"{output_dir}/{bloc_nom}"
            
            # Sauvegarder le bloc
            bloc.save(bloc_chemin)
            print(f"Bloc enregistré : {bloc_chemin}")


def main():
    diviser_image_en_blocs(
        image_path="ouestcharlie.png",
        taille_bloc=(640, 640),
        output_dir="blocs_output"
    )

    model = YOLO('./model-waldone.pt')    

    for f in Path('./blocs_output').iterdir():
        results = model(str(f))
        image_with_boxes = results[0].plot()
        image = Image.fromarray(image_with_boxes)
        image.show()

if __name__ == "__main__":
    main()