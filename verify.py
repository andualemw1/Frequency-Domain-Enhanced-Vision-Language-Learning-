
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_model.eval()

img_encoder_model = img_model.to(device)

# check parameters
print(next(img_encoder_model.parameters()).device)


img_batch = next(iter(img_loader))

img_batch = img_batch.to(device)


with torch.no_grad():

    img_embedding = img_encoder_model.forward_features(img_batch)


print(img_embedding.shape)


text_batch = next(iter(text_loader))


input_ids = text_batch["input_ids"].to(device)
attention_mask = text_batch["attention_mask"].to(device)


with torch.no_grad():

    text_outputs = text_encoder_model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )


text_embedding = text_outputs.last_hidden_state


print(text_embedding.shape)


