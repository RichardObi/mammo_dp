import logging
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from annoy import AnnoyIndex

from gan_compare.constants import get_classifier

###### Inference ######

def get_nn_dataset(initial_dataset, nn_indices):
    if len(nn_indices) == 2:
        # Removing the distances (at index position 1) from nn_indices tuple and retain only the indices (at index position 0)
        nn_indices = nn_indices[0]
    try:
        initial_dataset.remove(remaining_indices=nn_indices)
    except Exception as e:
        logging.warning(f"Now using fallback due to error while removing data (indices={nn_indices}) from initial dataset: {e}")
        initial_dataset = [data for i, data in enumerate(initial_dataset) if i in nn_indices] # this is an expensive operation.
    assert (len(nn_indices) == len(
        initial_dataset)), f"{len(nn_indices)} nn_indices should correspond to the new dataset size of {len(initial_dataset)}"
    return initial_dataset


def get_nn_dataloader(initial_dataset, nn_indices, batch_size=8, workers=2, shuffle=False, initial_dataloader=None, initial_metadata=None, replace_dataset_in_dataloader=True):
    logging.info(f"nn_indices: {nn_indices}")
    if replace_dataset_in_dataloader and initial_dataloader is not None and initial_metadata is not None:
        # remove indices from dataloader rather than initializing new one for quicker processing
        if len(nn_indices) == 2: nn_indices = nn_indices[0]
        logging.debug(f"initial_dataloader.dataset length a: {len(initial_dataloader.dataset)} ")
        initial_dataloader.dataset.replace(metadata=initial_metadata) # resetting dataset in dataloader
        logging.debug(f"initial_dataloader.dataset length b: {len(initial_dataloader.dataset)} ")
        initial_dataloader.dataset.remove(remaining_indices=nn_indices) # adjusting dataset in dataloader
        logging.debug(f"initial_dataloader.dataset length c: {len(initial_dataloader.dataset)} ")
        return initial_dataloader
    else:
        dataset = get_nn_dataset(initial_dataset, nn_indices)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            shuffle=shuffle,
        )


def get_top_k_from_image(image, model=None, embedding_model_type='inceptionv3', n=10, len_feature_vector=None,
                         distance_metric='angular', embedding_index=None, filename='val.ann', include_distances=True,
                         device=None, transforms=None, is_normalized= True,):
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    if len_feature_vector is None:
        len_feature_vector = get_len_feature_vector(embedding_model_type=embedding_model_type)
    if model is None:
        model = load_embedding_model(embedding_model_type=embedding_model_type, device=device)
    if transforms is None:
        transforms = get_transforms(embedding_model_type=embedding_model_type, is_normalized=is_normalized)
    embedded_image_batch = embed(model=model, image_batch=image, device=device, transforms=transforms)
    embedded_image = embedded_image_batch[0]
    logging.debug(f"Image was embedded using model {model.__class__.__name__} resulting in shape: {embedded_image.shape}")
    return find_top_k_nearest_neighbors(input_embedding=embedded_image, n=n, len_feature_vector=len_feature_vector,
                                        distance_metric=distance_metric, filename=filename,
                                        embedding_index=embedding_index, include_distances=include_distances)


def find_top_k_nearest_neighbors(input_embedding, n=10, len_feature_vector=2048, distance_metric='angular',
                                 filename='val.ann', embedding_index=None, include_distances=True):
    if embedding_index is None:
        embedding_index = AnnoyIndex(len_feature_vector, distance_metric)
        embedding_index.load(filename)  # super fast, will just mmap the file
    # let's find the n nearest neighbors in index by vector (not by item as our vector is not yet in the index)
    # top_k = embedding_index.get_nns_by_item(i=input_embedding, n=n, include_distances=include_distances)
    top_k = embedding_index.get_nns_by_vector(vector=input_embedding, n=n, include_distances=include_distances)
    return top_k


###### Indexing #######

def get_index_from_dataloader(my_dataloader, model=None, embedding_model_type='inceptionv3', distance_metric='angular',
                              num_trees=10, filename='val.ann', device=None, transforms=None, is_normalized=True):
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    len_feature_vector = get_len_feature_vector(embedding_model_type=embedding_model_type)
    if model is None:
        model = load_embedding_model(embedding_model_type=embedding_model_type, device=device)
    if transforms is None:
        my_dataloader.dataset.transforms = get_transforms(embedding_model_type=embedding_model_type, is_normalized=is_normalized)
        logging.debug(f"Changed my_dataloader.dataset.transforms to {my_dataloader.dataset.transforms}")
        if hasattr(my_dataloader.dataset, 'final_shape') and (embedding_model_type=='inceptionv3' or embedding_model_type=='vgg16'):
            my_dataloader.dataset.final_shape = (299, 299)
    embedding_list = embed_dataset(model, my_dataloader, device)
    embedding_index = create_embedding_index(embedding_list=embedding_list, len_feature_vector=len_feature_vector,
                                             distance_metric=distance_metric, num_trees=num_trees, filename=filename)
    return embedding_index


def create_embedding_index(embedding_list, len_feature_vector=2048, distance_metric='angular', num_trees=10,
                           filename='val.ann'):
    embedding_index = AnnoyIndex(len_feature_vector, distance_metric)
    for i, embedding in enumerate(embedding_list):
        embedding_index.add_item(i, embedding)
    embedding_index.build(num_trees)  # e.g. 10 trees
    embedding_index.save(filename)
    return embedding_index


def embed_dataset(model, my_dataloader, device):
    embedding_list = []
    for i, data in enumerate(tqdm(my_dataloader)):
        # get the inputs; data is a list of [inputs, labels]
        try:
            samples, labels, _, _, _, _ = data
        except:
            samples, labels, _, _, _ = data
        # embed each sample image
        logging.debug(f"Now trying to embed image batch of shape {samples.shape}")
        embeddings = embed(model=model, image_batch=samples, device=device).tolist()
        embedding_list.extend(embeddings)
    return embedding_list


def embed(model, image_batch, device, transforms=None, to_numpy=True):
    # preprocess if need be
    if transforms is not None:
        image_batch = transforms(image_batch)
    # Pass input images through embedding model
    with torch.no_grad():
        embedded_batch = model(image_batch.to(device))
    if to_numpy:
        embedded_batch = embedded_batch.detach().cpu().numpy()
    return embedded_batch


def load_embedding_model(embedding_model_type='inceptionv3', device=None, progress=True, activate_dropout=False, config=None, embeddings_from_last_layer_minus: int = 2):
    if embedding_model_type == 'inceptionv3':
        # Load Inception v3 model
        model = models.inception_v3(pretrained=True, progress=progress)
    elif embedding_model_type == 'vgg16':
        # Load VGG16 model
        model = models.vgg16(pretrained=True, progress=progress)
    elif embedding_model_type == 'resnet50':
        # Load ResNet50 model
        model = models.resnet50(pretrained=True, progress=progress)
    elif ('swin' in embedding_model_type or embedding_model_type == 'config'): # get model from config
        # Load Swin Transformer model
        model = get_classifier(config=config, device=device)
    else:
        raise Exception(
            f"The embedding model type you provided ('{embedding_model_type}') is not available. For now, please choose between 'inceptionv3', 'vgg16', and 'resnet50' or extend the implementation to further models.")

    # e.g., remove last fully-connected layer (and maybe other layers such as dropout layers)
    #model = torch.nn.Sequential(*(list(model.children())[:-embeddings_from_last_layer_minus]))
    if hasattr(model, 'head'):
        # removing the prediction 'head' layer
        model.head = torch.nn.Identity()
    elif hasattr(model, 'fc'): # elif instead of if as (swin) transformers might have layers called 'fc' inside the transformer block
        # removing the last fully connected layer
        model.fc = torch.nn.Identity()
    if activate_dropout and hasattr(model, 'dropout'):
        # removing dropout layer to get more reproducible results
        model.dropout = torch.nn.Identity()
    model.to(device).eval()
    return model


def get_transforms(is_normalized=True, embedding_model_type='inceptionv3'):
    if embedding_model_type == 'inceptionv3':
        if is_normalized:
            return transforms.Compose([
                transforms.Resize(299),  # Resize image to 299x299 pixels
                #transforms.ToTensor(),  # Convert image to tensor
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize image
            ])
        else:
            return transforms.Compose([
                transforms.Resize(299),  # Resize image to 299x299 pixels
                #transforms.ToTensor(),  # Convert image to tensor
            ])
    elif embedding_model_type == 'vgg16':
        if is_normalized:
            return transforms.Compose([
                transforms.Resize(299),  # Resize image to 299x299 pixels
                #transforms.ToTensor(),  # Convert image to tensor
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize image
            ])
        else:
            return transforms.Compose([
                transforms.Resize(299),  # Resize image to 299x299 pixels
                #transforms.ToTensor(),  # Convert image to tensor
            ])
    elif embedding_model_type == 'resnet50':
        if is_normalized:
            return transforms.Compose([
                transforms.Resize(224),  # Resize image to 224x224 pixels
                #transforms.ToTensor(),  # Convert image to tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
            ])
        else:
            return transforms.Compose([
                transforms.Resize(224),  # Resize image to 224x224 pixels
                #transforms.ToTensor(),  # Convert image to tensor
            ])
    elif ('swin' in embedding_model_type or embedding_model_type == 'config'):
        if is_normalized:
            return transforms.Compose([
                transforms.Resize(224),  # Resize image to 224x224 pixels
                #transforms.ToTensor(),  # Convert image to tensor
                transforms.Normalize((0.5), (0.5))
            ])
        else:
            return transforms.Compose([
                transforms.Resize(224),  # Resize image to 224x224 pixels
                #transforms.ToTensor(),  # Convert image to tensor
            ])
    else:
        raise Exception(
            f"The embedding model type you provided ('{embedding_model_type}') is not available. For now, please choose between 'inceptionv3', 'vgg16', and 'resnet50' or extend the implementation to further models.")


def get_len_feature_vector(embedding_model_type='inceptionv3'):
    # TODO debug and check if dims are correct.
    if embedding_model_type == 'inceptionv3':
        return 2048
    elif embedding_model_type == 'vgg16':
        return 4096
    elif embedding_model_type == 'resnet50':
        return 2048
    elif ('swin' in embedding_model_type or embedding_model_type == 'config'):
        # swin-t transformer feature extractor returns 768 dimensional vectors
        return 768
    else:
        raise Exception(
            f"The embedding model type you provided ('{embedding_model_type}') is not available. For now, please choose between 'inceptionv3', 'vgg16', and 'resnet50' or extend the implementation to further models.")
