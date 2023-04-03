from io import BytesIO
from PIL import Image
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper, FileLister
from typing import Dict, List, Optional, Union


def image_decode(file_handle, image_format: str = 'RGB'):
    """
    Decode image and convert to specific `image_format`.
    """
    with BytesIO(file_handle.read()) as b:
        img = Image.open(b)
        img.load()
    if image_format:
        img = img.convert(image_format)
    return img


def decode(item):
    """
    Decode the item based on file type. In this case, we have `.cls` and `.jpg` files.
    """
    key, value = item
    if key.endswith(".cls"):
        return key, int(value.read().decode("utf-8"))
    if key.endswith(".jpg"):
        return key, image_decode(value)


def select(web_dataset_dict: Dict, img_key: str = '.jpg', target_key: str = '.cls'):
    """
    Given a `Dict` object, return a tuple of `(image, target_label)`.
    """
    return web_dataset_dict[img_key], web_dataset_dict[target_key]


def create_files_datapipe(paths: List[str], file_system: str = "local"):
    """
    Given a file system and paths, returns an IterDataPipe of file handlers.

    `.sharding_filter()` will shard the workload across nodes and workers within each node.
    In this case, each worker will read a different set of tar archives.
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
        # download: bool = False,  # Not used for now
        # batch_size: Optional[int] = None,  # Batch size will be defined later while creating `DataLoader2`
        # seed: int = 42,  # Seed will be set later by `DataLoader2`
        repeats: int = 0,
        **kwargs
) -> IterDataPipe:
    r"""
    Given a `root` and `file_system`, create a datapipe line that finds the corresponding files and decode them.

    Directory set up as follows:
        ./train
            imagenet-train-00001.tar
            imagenet-train-00002.tar
            imagenet-train-00003.tar
        ./valid
            imagenet-val-00001.tar
            imagenet-val-00002.tar
            imagenet-val-00003.tar
    """
    if isinstance(root, str):
        root = [root]

    if file_system == "local":
        root = [path + f"/{split}/"for path in root]

    tar_archives_dp = create_files_datapipe(root, file_system)

    # Note: after `.map(decode)`, we have an iterable of decoded files that were stored inside the tar archives.
    #   e.g. `00001000.cls`, `00001000.jpg`, `00001001.cls`, `00001001.jpg`, ...
    # The built-in `.webdataset()` DataPipe groups the corresponding files into a `Dict`,
    # such that you will have one `Dict` for each sample (i.e. one for `00001000`, one for `00001001`).
    datapipe = tar_archives_dp.load_from_tar() \
                   .map(decode) \
                   .webdataset() \
                   .map(select)

    if repeats > 0:
        datapipe.cycle(repeats)

    return datapipe
