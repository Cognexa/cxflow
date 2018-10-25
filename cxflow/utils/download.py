import click
import logging
import os
import requests
import shutil
import re


def sanitize_url(url: str) -> str:
    """
    Sanitize the given url so that it can be used as a valid filename.

    :param url: url to create filename from
    :raise ValueError: when the given url can not be sanitized
    :return: created filename
    """

    for part in reversed(url.split('/')):
        filename = re.sub(r'[^a-zA-Z0-9_.\-]', '', part)
        if len(filename) > 0:
            break
    else:
        raise ValueError('Could not create reasonable name for file from url %s', url)

    return filename


def maybe_download_and_extract(data_root: str, url: str) -> None:
    """
    Maybe download the specified file to ``data_root`` and try to unpack it with ``shutil.unpack_archive``.
    
    :param data_root: data root to download the files to
    :param url: url to download from
    """

    # make sure data_root exists
    os.makedirs(data_root, exist_ok=True)

    # create sanitized filename from url
    filename = sanitize_url(url)

    # check whether the archive already exists
    filepath = os.path.join(data_root, filename)
    if os.path.exists(filepath):
        logging.info('\t`%s` already exists; skipping', filepath)
        return

    # download with progressbar
    try:
        logging.info('\tdownloading %s', filepath)
        req = requests.get(url, stream=True)
        req.raise_for_status()
    except requests.exceptions.RequestException as ex:
        logging.error('File `%s` could not be downloaded, %s', filepath, ex)
        return

    expected_size = int(req.headers.get('content-length'))
    chunk_size = 1024
    with open(filepath, 'wb') as f_out,\
         click.progressbar(req.iter_content(chunk_size=chunk_size), length=expected_size/chunk_size) as bar:
        for chunk in bar:
            if chunk:
                f_out.write(chunk)
                f_out.flush()

    # extract
    try:
        shutil.unpack_archive(filepath, data_root)
    except (shutil.ReadError, ValueError):
        logging.info('File `%s` could not be extracted by `shutil.unpack_archive`. Please process it manually.',
                     filepath)
