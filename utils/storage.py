import os.path

from libcloud.storage.base import Container, StorageDriver
from libcloud.storage.drivers.local import LocalStorageDriver
from libcloud.storage.drivers.google_storage import GoogleStorageDriver
from libcloud.storage.drivers.azure_blobs import AzureBlobsStorageDriver
from libcloud.storage.drivers.s3 import S3StorageDriver

from utils.concurrency import run_in_executor
import utils.environment as env

buffer_size = 2 ** 20


def create_iter(container: Container, file_path: str, file_name: str):
    with open(file_path, mode='rb') as f:
        return container.upload_object_via_stream(iter(lambda: f.read(buffer_size), b""), file_name)


@run_in_executor
def a_create_iter(container: Container, file_path: str, file_name: str):
    return create_iter(container, file_path, file_name)



# Cloudlib storage driver
def get_local_driver():
    path = env.str('STORAGE_LOCAL_PATH', default='./dataset/res')
    return LocalStorageDriver(path)

def get_s3_driver():
    key = env.str('STORAGE_S3_KEY', error_msg='{0}: S3 storage key not found!')
    secret = env.str('STORAGE_S3_SECRET', error_msg='{0}: S3 secret not found!')
    secure = env.bool('STORAGE_S3_USE_HTTPS', default=True)
    host = env.str('STORAGE_S3_HOST')
    port = env.int('STORAGE_S3_PORT')
    return S3StorageDriver(key, secret, secure, host, port)

def get_google_cloud_driver():
    key = env.str('STORAGE_GOOGLE_STORAGE_KEY', error_msg='{0}: Google storage key not found!')
    secret = env.str('STORAGE_GOOGLE_STORAGE_SECRET', error_msg='{0}: Google storage secret not found!')
    project = env.str('STORAGE_GOOGLE_STORAGE_PROJECT', default=None)
    return GoogleStorageDriver(key, secret, project)

def get_azure_driver():
    key = env.str('STORAGE_AZURE_KEY', error_msg='{0}: Azure storage key not found!')
    secret = env.str('STORAGE_AZURE_SECRET', error_msg='{0}: Azure storage secret not found!')
    secure = env.bool('STORAGE_AZURE_USE_HTTPS', default=True)
    host = env.str('STORAGE_AZURE_HOST')
    port = env.int('STORAGE_AZURE_PORT')
    return AzureBlobsStorageDriver(key, secret, secure, host, port)


def get_storage_provider() -> StorageDriver:
    """
    Gets the storage driver to be used.
    The storage driver to be used is determined by the value of the STORAGE_DRIVER environment variable.
    :return: StorageDriver - The storage driver to be used.
    """

    # Get storage driver
    drivers = {
        'LOCAL': get_local_driver,
        'S3': get_s3_driver,
        'GOOGLE_CLOUD': get_google_cloud_driver,
        'AZURE': get_azure_driver,
    }
    driver_name = env.enum('STORAGE_TYPE', list(drivers.keys()), default='LOCAL')
    return drivers[driver_name]()

def get_storage_container(name: str) -> Container:
    """
    Gets the container object associated with the given name.
    If a container with the given name does not exist, it is created.
    If a container cannot be created, an error is raised.
    The storage driver to be used is determined by the value of the STORAGE_DRIVER environment variable.

    :param name The name of the container to get or create.
    :return: A container object associated with the given name.
    """

    driver = get_storage_provider()
    if not name in [it.name for it in driver.list_containers()]:
        driver.create_container(name)
    return driver.get_container(name)
