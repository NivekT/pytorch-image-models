from io import BytesIO
from PIL import Image
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper, FileLister
from typing import Dict, List, Optional, Union


def image_decode(file_handle, image_format: str = 'RGB'):
    with BytesIO(file_handle.read()) as b:
        img = Image.open(b)
        img.load()
    if image_format:
        img = img.convert(image_format)
    return img


def decode(item):
    key, value = item
    if key.endswith(".cls"):
        return key, int(value.read().decode("utf-8"))
    if key.endswith(".jpg"):
        return key, image_decode(value)


def select(web_dataset_dict: Dict, img_key: str = '.jpg', target_key: str = '.cls'):
    return web_dataset_dict[img_key], web_dataset_dict[target_key]


def create_files_datapipe(paths: List[str], file_system: str = "local"):
    """
    Given a file system and paths, returns an IterDataPipe of file handlers.
    """
    paths_dp = IterableWrapper(paths)
    if file_system == "local":
        datapipe = paths_dp.list_files(masks="*.tar") \
                        .shuffle() \
                        .sharding_filter() \
                        .open_files(mode="b")
    elif file_system == "s3":
        datapipe = paths_dp.list_files_by_s3(masks="*.tar") \
                        .shuffle() \
                        .sharding_filter() \
                        .load_files_by_s3()
    else:
        raise NotImplementedError(f"Not implemented for file_system {file_system} yet.")

    return datapipe


def create_datapipe(
        name: str,
        root: Union[str, List[str]],
        split: str,
        file_system: str = "local",
        download: bool = False,
        batch_size: Optional[int] = None,
        seed: int = 42,
        repeats: int = 0,
        **kwargs
) -> IterDataPipe:
    r"""

    """
    if isinstance(root, str):
        root = [root]

    if file_system == "local":
        root = [path + f"/{split}/"for path in root]

    files_datapipe = create_files_datapipe(root, file_system)
    datapipe = files_datapipe.load_from_tar() \
                   .map(decode) \
                   .webdataset() \
                   .map(select)

    return datapipe
