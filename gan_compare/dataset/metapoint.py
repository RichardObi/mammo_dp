from dataclasses import dataclass, field
from typing import List, Optional, Union

from gan_compare.constants import DATASET_LIST
from gan_compare.dataset.constants import (
    BIOPSY_STATUS,
    BIRADS_DICT,
    DENSITY_DICT,
    LATERALITIES,
    ROI_TYPES,
    VIEWS,
)
from gan_compare.paths import DATASET_PATH_DICT


@dataclass
class Metapoint:
    patch_id: int
    patient_id: str
    image_id: Union[str, int]
    laterality: str
    view: str
    image_path: str
    dataset: str
    bbox: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    roi_type: List[str] = field(default_factory=lambda: ["healthy"])
    biopsy_proven_status: str = "unknown"
    birads: str = "0"
    density: int = -1
    predefined_set: Optional[str] = None
    lesion_id: Optional[str] = None
    contour: Optional[List[List[int]]] = None
    radiomics: Optional[dict] = None
    subtlety: int = -1 #'unknown' # -1 is okay as long as there is no -1 in none of the datasets for subtlety somewhere.
    mass_shape: str = 'unknown'
    mass_margins: str = 'unknown'

    def __post_init__(self):
        if type(self.roi_type) == str:
            self.roi_type = [self.roi_type]
        if self.biopsy_proven_status is None:
            self.biopsy_proven_status = "unknown"
        self.image_id = str(self.image_id)
        self.patient_id = str(self.patient_id)
        self.predefined_set = self.predefined_set
        if not self.patient_id.startswith(tuple(DATASET_LIST)):
            self.patient_id = self.dataset + "__" + self.patient_id
        self.birads = str(self.birads)
        self.biopsy_proven_status = self.biopsy_proven_status.lower()
        if self.biopsy_proven_status == "malign":
            self.biopsy_proven_status = "malignant"
        self.density = int(self.density)
        assert (
            "healthy" not in self.roi_type or len(self.roi_type) == 1
        ), "If roi type is healthy it cannot be any other type"
        assert all(
            roi_type in ROI_TYPES for roi_type in self.roi_type
        ), f"{self.roi_type} not in known ROI_TYPES"
        assert len(self.bbox) == 4, "Wrong bbox format"
        assert (
            self.laterality in LATERALITIES
        ), f"{self.laterality} not in known LATERALITIES"
        assert self.view in VIEWS, f"{self.view} not in known VIEWS"
        assert (
            self.biopsy_proven_status in BIOPSY_STATUS
        ), f"{self.biopsy_proven_status} not in known BIOPSY_STATUS"
        assert (
            self.birads in BIRADS_DICT.keys() or self.birads == "0"
        ), f"{self.birads} is not a known birads value"
        assert (
            self.density in DENSITY_DICT.keys() or self.density == -1
        ), f"{self.density} not in known DENSITY_DICT"
        self.is_healthy = "healthy" in self.roi_type
        self.is_benign = (
            self.biopsy_proven_status == "benign" or self.biopsy_proven_status == "benign_without_callback"
            if self.biopsy_proven_status in ["benign", "benign_without_callback", "malignant"]
            else -1
        )  # this will throw an error later if benign/malignant classification is attempted and there is a metapoint without a biopsy_proven_status

    def get_absolute_image_path(self) -> str:
        return str((DATASET_PATH_DICT[self.dataset] / self.image_path).resolve())
