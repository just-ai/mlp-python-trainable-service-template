import pickle
from pathlib import Path
from typing import Union, Type

from mlp_sdk.abstract import Task, LearnableMixin
from mlp_sdk.hosting.host import host_mlp_cloud
from mlp_sdk.log import get_logger
from mlp_sdk.storage.local_storage import LocalStorage
from mlp_sdk.storage.s3_storage import S3Storage
from mlp_sdk.transport.MlpServiceSDK import MlpServiceSDK
from mlp_sdk.types import ItemsCollection, TextsCollection, Items, Item, ServiceInfo, DatasetInfo
from mlp_sdk.utilities.misc import get_env
from pydantic import BaseModel

LOGGER = get_logger(__name__)


class FitActionExample(Task, LearnableMixin):

    def __init__(self, config: BaseModel, service_sdk: MlpServiceSDK = None) -> None:
        super().__init__(config, service_sdk)

        self.model = FittedMLModel(dict())
        self.storage = self._get_storage()
        self.model_path = 'model.pkl'

        self.is_fitted_model = False

        try:
            self._load_state()
        except KeyError as e:
            LOGGER.error(f'Unable to load saved state, error message: {str(e)}')

    @property
    def is_fitted(self):
        return self.is_fitted_model

    def fit(
            self,
            train_data: TextsCollection,
            targets: BaseModel,
            config: BaseModel,
            target_service_info: ServiceInfo,
            dataset_info: DatasetInfo,
            model_dir: str,
            previous_model_dir: str,
    ) -> None:
        self.storage = self._get_storage(model_dir)

        try:
            self.model = FittedMLModel(self._prepare_model_data(train_data.texts))
            self._save_state()
            self.is_fitted_model = True
        except Exception as e:
            LOGGER.error(f"Fit execution error: {e}")

    def predict(self, data: TextsCollection, config: BaseModel) -> ItemsCollection:
        result_list = []
        for text in data.texts:
            if not self.model.has_data(text):
                raise ValueError('No such id here, try 0 or 1')

            predict_result = self.model.predict(text)
            item_list = [Item(value=str(predict_result))]
            result_list.append(Items(items=item_list))

        return ItemsCollection(items_list=result_list)

    def _save_state(self):
        with self.storage.open(self.model_path, 'wb') as fout:
            pickle.dump(self.model, fout)

    def _load_state(self):
        with self.storage.open(self.model_path, 'rb') as fin:
            self.model = pickle.loads(fin.read())
        self.is_fitted_model = True

    def prune_state(self, model_dir: str = '') -> None:
        remove_path = model_dir if len(model_dir) > 0 else get_env('MLP_STORAGE_DIR')
        storage = self._get_storage(remove_path)
        storage.remove(self.model_path)

    @staticmethod
    def _get_storage(model_dir: str = '') -> Union[LocalStorage, S3Storage]:
        storage_type = get_env('MLP_STORAGE_TYPE')
        storage_dir = get_env('MLP_STORAGE_DIR') if len(model_dir) == 0 else model_dir

        if storage_type == LocalStorage.name():
            storage = LocalStorage(path=Path(storage_dir))

        elif storage_type == S3Storage.name():
            storage = S3Storage(
                bucket=get_env('MLP_S3_BUCKET'),
                data_dir=storage_dir,
                service_name='s3',
                region=get_env('MLP_S3_REGION', ''),
                access_key=get_env('MLP_S3_ACCESS_KEY'),
                secret_key=get_env('MLP_S3_SECRET_KEY'),
                endpoint=get_env('MLP_S3_ENDPOINT'),
            )

        else:
            message = f"storage_type {storage_type} is invalid."
            raise ValueError(message)

        return storage

    @staticmethod
    def _prepare_model_data(texts: [str]):
        data = dict()
        for idx, text in enumerate(texts):
            data[str(idx)] = text
        return data


class FittedMLModel:

    def __init__(self, data):
        self.data = data

    def has_data(self, index):
        return index in self.data

    def predict(self, index):
        if index in self.data:
            return self.data[str(index)]
        else:
            return "No such sentence in the original dataset"


if __name__ == "__main__":
    host_mlp_cloud(FitActionExample, BaseModel())
