# 数据生成器，数据集的格式为coco的格式，可以直接使用代码将


class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, class_id, class_name):

        # Does the class exist already?
        for info in self.class_info:
            if info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)  # 总共多少类
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)  # 多少个图片
        print(self.num_images)
        self._image_ids = np.arange(self.num_images)

        self.class_from_source_map = {"{}".format(info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}  # 做好类别名称的映射

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

        print(self.class_from_source_map)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        # if source_class_id in self.class_from_source_map.keys():
        #   return self.class_from_source_map[source_class_id]
        # else:
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    def append_data(self, class_info, image_info):
        self.external_to_class_id = {}
        for i, c in enumerate(self.class_info):
            for ds, id in c["map"]:
                self.external_to_class_id[ds + str(id)] = i

        # Map external image IDs to internal ones.
        self.external_to_image_id = {}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info["ds"] + str(info["id"])] = i

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's availble online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


class CocoDataset(Dataset):
    pass
    # def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
    #               class_map=None, return_coco=False, auto_download=False):
    #     """Load a subset of the COCO dataset.
    #     dataset_dir: The root directory of the COCO dataset.
    #     subset: What to load (train, val, minival, valminusminival)
    #     year: What dataset year to load (2014, 2017) as a string, not an integer
    #     class_ids: If provided, only loads images that have the given classes.
    #     class_map: TODO: Not implemented yet. Supports maping classes from
    #         different datasets to the same class ID.
    #     return_coco: If True, returns the COCO object.
    #     auto_download: Automatically download and unzip MS-COCO images and annotations
    #     """
    #
    #     if auto_download is True:
    #         self.auto_download(dataset_dir, subset, year)
    #
    #     coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))  # 载入coco数据集合json文件
    #     if subset == "minival" or subset == "valminusminival":
    #         subset = "val"
    #     image_dir = "{}/{}{}".format(dataset_dir, subset, year)
    #
    #     # Load all classes or a subset?
    #     if not class_ids:
    #         # All classes
    #         class_ids = sorted(coco.getCatIds())
    #
    #     # All images or a subset?
    #     if class_ids:
    #         image_ids = []  # 找到包含所有需要id的图片
    #         for id in class_ids:
    #             image_ids.extend(list(coco.getImgIds(catIds=[id])))
    #         # Remove duplicates
    #         image_ids = list(set(image_ids))
    #     else:
    #         # All images
    #         image_ids = list(coco.imgs.keys())
    #
    #     # Add classes
    #     for i in class_ids:  # 添加类别信息 格式  [{"source": "", "id": 0, "name": "BG"}]
    #         self.add_class(i, coco.loadCats(i)[0]["name"])
    #
    #     # Add images
    #     for num, i in enumerate(image_ids):
    #         if num <= 200:
    #             self.add_image(
    #                 image_id=i,
    #                 path=os.path.join(image_dir, coco.imgs[i]['file_name']),
    #                 width=coco.imgs[i]["width"],
    #                 height=coco.imgs[i]["height"],
    #                 annotations=coco.loadAnns(coco.getAnnIds(
    #                     imgIds=[i], catIds=class_ids, iscrowd=None)))
    #     if return_coco:
    #         return coco


class OwnDataset(CocoDataset):
    def load_own(self, json_path, image_dir, return_coco=False):
        coco = COCO(json_path)
        class_ids = sorted(coco.getCatIds())
        image_ids = list(coco.imgs.keys())
        for i in class_ids:  # 添加类别信息 格式  [{"source": "", "id": 0, "name": "BG"}]
            self.add_class(i, coco.loadCats(i)[0]["name"])

        for i in image_ids:
            self.add_image(
                image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco
