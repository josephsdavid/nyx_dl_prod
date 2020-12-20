import numpy as np
import torch
from lightning import AE as AutoencoderModel
import pickle
from typing import Dict, Union, List, Tuple

MovieMapping = Dict[int, Dict[str, Union[str, int]]]
# keys are values
InputDict = Dict[str, float]


def load_model(path: str = "final_mod.ckpt", device: str="cpu") -> AutoencoderModel:
    model = AutoencoderModel.load_from_checkpoint(path, device_map=device)
    return model

def load_mapping(path: str = "movie_map.pkl") -> MovieMapping:
    with open(path, 'rb') as f:
        return pickle.load(f)

def title_to_index(title: str, mapping: MovieMapping) -> int:
    title_index_dict = {v['title']: v['index'] for v in mapping.values()}
    return title_index_dict[title]


def input_to_vector(input: InputDict, mapping: MovieMapping) -> Tuple[torch.tensor, List[int]]:
    assert max(input.values()) <= 5
    vector = torch.zeros(28038)
    indices = []
    for k, v in input.items():
        index = title_to_index(k, mapping)
        indices.append(index)
        vector[index] = v
    return vector, indices


def predict_vector(v: torch.tensor, model: AutoencoderModel) -> np.ndarray:
    v.to(model.device)
    model.eval()
    with torch.no_grad():
        v = v / 5
        result = model(v)['predicted']

    result = result.detach().cpu().numpy()
    # rescale to be from 0 to 5
    return result

def get_top_n(v: np.ndarray, index_list: List[int], n: int) -> List[int]:
    v[np.array(index_list)] = 0
    return list(np.argpartition(-v, n)[:n])


def main(inputs: InputDict) -> List[int]:
    mapping = load_mapping()
    model = load_model()
    x, indices = input_to_vector(inputs, mapping)
    preds = predict_vector(x, model)
    import pdb;pbd.set_trace()
    top_n = get_top_n(preds, indices, 5)
    return list(top_n)


if __name__ == "__main__":
    # TODO: if movie not in movie mapping list of movies, return error (only popular movies are here)
    movies = [v['title']  for v in load_mapping().values()][::3][-40:]
    scores = list(np.clip( 1., 6., (abs(np.random.rand(40)) * 5)))
    input = dict(zip(movies, scores))
    result = main(input)
    print("ratings")
    print(input)
    print("recomendations:")
    print(result)
    import pdb; pdb.set_trace()
