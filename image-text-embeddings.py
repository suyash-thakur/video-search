from imgBind import data
import torch
from imgBind.models import imagebind_model
from imgBind.models.imagebind_model import ModalityType

text_list = ["Kendrick lamar on white bed"]

image_paths = ["./img/1.png"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

print("Model loaded")

def get_image_embeddings(image_paths):
    
    inputs = {
        ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    }

    with torch.no_grad():
        embeddings = model(inputs)

    return embeddings

def get_text_embeddings(text_list):
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    }

    with torch.no_grad():
        embeddings = model(inputs)

    return embeddings


image_embeddings = get_image_embeddings(image_paths)
text_embeddings = get_text_embeddings(text_list)
print("Embedding Generation completed")

print(text_embeddings)
print(image_embeddings)

print(
    "Vision x Text: ",
    torch.softmax(image_embeddings[ModalityType.VISION] @ text_embeddings[ModalityType.TEXT].T, dim=-1),
)

## Find the most similar text to the image
